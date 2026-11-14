import { AudioCapture } from "./capture";
import {
  OnlineSpeakerClusterer,
  REJECT_LABELS,
  type ClusterDecision,
  type RejectReason,
} from "./cluster";
import {
  SAMPLE_RATE,
  classifyAudio,
  extractEmbedding,
  type AudioClass,
  type VoiceEmbedding,
} from "./features";
import { loadVoiceprint, saveVoiceprint, clearVoiceprint, type StoredVoiceprint } from "./storage";
import { detectSpeechSegments } from "./vad";
import { rms } from "./math";

export type EngineStatus = "idle" | "enrolling" | "listening" | "error";

export type LiveEvent = {
  at: number;
  gate: AudioClass;
  decision: ClusterDecision;
  level: number;
};

export type EngineSnapshot = {
  status: EngineStatus;
  enrolledCount: number;
  meThreshold: number;
  otherThreshold: number;
  peopleEstimate: number;
  lastGate: AudioClass | null;
  lastSeatLabel: string | null;
  lastRejectLabel: string | null;
  rejectHistory: string[];
  level: number;
  error: string | null;
};

export type EngineListener = (snap: EngineSnapshot) => void;

const ENROLL_TARGET = 3;
const ANALYSIS_WINDOW = SAMPLE_RATE; // ~1s

/**
 * High-level 听席 engine: enroll → live listen → gate → cluster → UI snapshot.
 */
export class TingxiEngine {
  private clusterer = new OnlineSpeakerClusterer();
  private capture: AudioCapture | null = null;
  private status: EngineStatus = "idle";
  private enrollments: VoiceEmbedding[] = [];
  private buffer: number[] = [];
  private lastGate: AudioClass | null = null;
  private lastSeatLabel: string | null = null;
  private lastRejectLabel: string | null = null;
  private rejectHistory: string[] = [];
  private level = 0;
  private error: string | null = null;
  private listeners = new Set<EngineListener>();
  private createdAt: string | undefined;

  constructor() {
    this.hydrate();
  }

  private hydrate(): void {
    const stored = loadVoiceprint();
    if (!stored) return;
    this.createdAt = stored.createdAt;
    this.enrollments = stored.enrollments;
    this.clusterer.setThresholds(stored.meThreshold, stored.otherThreshold);
    if (this.enrollments.length > 0) {
      this.clusterer.enrollMe(this.enrollments);
    }
  }

  subscribe(listener: EngineListener): () => void {
    this.listeners.add(listener);
    listener(this.snapshot());
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    const snap = this.snapshot();
    for (const l of this.listeners) l(snap);
  }

  snapshot(): EngineSnapshot {
    return {
      status: this.status,
      enrolledCount: this.enrollments.length,
      meThreshold: this.clusterer.meThreshold,
      otherThreshold: this.clusterer.otherThreshold,
      peopleEstimate: this.clusterer.peopleEstimate,
      lastGate: this.lastGate,
      lastSeatLabel: this.lastSeatLabel,
      lastRejectLabel: this.lastRejectLabel,
      rejectHistory: this.rejectHistory.slice(-12),
      level: this.level,
      error: this.error,
    };
  }

  setThresholds(me: number, other: number): void {
    this.clusterer.setThresholds(me, other);
    this.persist();
    this.emit();
  }

  private persist(): void {
    if (this.enrollments.length === 0) return;
    const saved: StoredVoiceprint = saveVoiceprint({
      createdAt: this.createdAt,
      enrollments: this.enrollments,
      meThreshold: this.clusterer.meThreshold,
      otherThreshold: this.clusterer.otherThreshold,
    });
    this.createdAt = saved.createdAt;
  }

  clearEnrollment(): void {
    this.enrollments = [];
    this.clusterer.reset();
    this.rejectHistory = [];
    this.lastSeatLabel = null;
    this.lastRejectLabel = null;
    this.createdAt = undefined;
    clearVoiceprint();
    this.emit();
  }

  /** Add one enrollment clip (need 3). Returns true when ready. */
  enrollFromSamples(samples: ArrayLike<number>): { ok: boolean; message: string } {
    const gate = classifyAudio(samples, SAMPLE_RATE);
    this.lastGate = gate.label;
    if (gate.label !== "speech") {
      const reason = REJECT_LABELS[gate.label as RejectReason] ?? "非语音";
      this.pushReject(reason);
      this.emit();
      return { ok: false, message: `录入失败：${reason}` };
    }
    const embedding = extractEmbedding(samples, SAMPLE_RATE);
    if (!embedding) {
      this.pushReject("无法提取声纹");
      this.emit();
      return { ok: false, message: "录入失败：无法提取声纹" };
    }
    this.enrollments = [...this.enrollments, embedding].slice(-ENROLL_TARGET);
    this.clusterer.enrollMe(this.enrollments);
    this.persist();
    this.emit();
    const done = this.enrollments.length >= ENROLL_TARGET;
    return {
      ok: true,
      message: done
        ? "声纹已就绪，可以开始听席"
        : `已录入 ${this.enrollments.length}/${ENROLL_TARGET} 段`,
    };
  }

  async startListening(): Promise<void> {
    if (this.enrollments.length < ENROLL_TARGET) {
      this.error = `请先录入 ${ENROLL_TARGET} 段声纹`;
      this.status = "error";
      this.emit();
      return;
    }
    if (this.status === "listening") return;

    this.clusterer.clearOthers();
    this.clusterer.enrollMe(this.enrollments);
    this.buffer = [];
    this.error = null;
    this.status = "listening";
    this.emit();

    this.capture = new AudioCapture({
      onFrame: (samples) => this.onFrame(samples),
      onError: (err) => {
        this.error = err.message;
        this.status = "error";
        this.emit();
      },
    });
    await this.capture.start();
  }

  async stopListening(): Promise<void> {
    await this.capture?.stop();
    this.capture = null;
    this.status = "idle";
    this.level = 0;
    this.emit();
  }

  private onFrame(samples: Float32Array): void {
    this.level = rms(samples);
    for (let i = 0; i < samples.length; i++) this.buffer.push(samples[i]!);
    // Keep ~2s
    const max = SAMPLE_RATE * 2;
    if (this.buffer.length > max) {
      this.buffer = this.buffer.slice(this.buffer.length - max);
    }
    if (this.buffer.length < ANALYSIS_WINDOW) {
      this.emit();
      return;
    }

    const window = Float32Array.from(this.buffer.slice(-ANALYSIS_WINDOW));
    const gate = classifyAudio(window, SAMPLE_RATE);
    this.lastGate = gate.label;

    if (gate.label !== "speech") {
      const decision = this.clusterer.assign(
        { vector: [], pitchHz: null, energyDb: gate.energyDb },
        gate.label,
      );
      this.applyDecision(decision);
      // Drop half buffer to avoid sticky rejects
      this.buffer = this.buffer.slice(Math.floor(ANALYSIS_WINDOW / 2));
      this.emit();
      return;
    }

    const segments = detectSpeechSegments(window);
    const target = segments.length > 0 ? segments[segments.length - 1]!.samples : window;
    const embedding = extractEmbedding(target, SAMPLE_RATE);
    if (!embedding) {
      this.pushReject(REJECT_LABELS.not_speech);
      this.lastSeatLabel = null;
      this.emit();
      return;
    }

    const decision = this.clusterer.assign(embedding, "speech");
    this.applyDecision(decision);
    this.buffer = this.buffer.slice(Math.floor(ANALYSIS_WINDOW / 2));
    this.emit();
  }

  private applyDecision(decision: ClusterDecision): void {
    this.lastSeatLabel = decision.seat?.label ?? null;
    this.lastRejectLabel = decision.rejectLabel;
    if (decision.rejectLabel) this.pushReject(decision.rejectLabel);
  }

  private pushReject(label: string): void {
    if (this.rejectHistory[this.rejectHistory.length - 1] === label) return;
    this.rejectHistory = [...this.rejectHistory, label].slice(-20);
  }
}

export { ENROLL_TARGET, REJECT_LABELS };
