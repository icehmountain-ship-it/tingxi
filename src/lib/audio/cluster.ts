import { embeddingSimilarity, type VoiceEmbedding } from "./features";
import { averageVectors, clamp, l2Normalize } from "./math";

export type SpeakerSeat =
  | { kind: "me"; label: "你" }
  | { kind: "other"; id: number; label: string };

export type RejectReason =
  | "silence"
  | "noise"
  | "music"
  | "not_speech"
  | "pitch_mismatch"
  | "below_threshold"
  | "no_enrollment";

export type ClusterDecision = {
  seat: SpeakerSeat | null;
  similarity: number;
  rejectReason: RejectReason | null;
  rejectLabel: string | null;
  peopleEstimate: number;
};

export type ClustererOptions = {
  meThreshold?: number;
  otherThreshold?: number;
  pitchRatioMin?: number;
  pitchRatioMax?: number;
  maxOthers?: number;
};

const OTHER_LABELS = ["说话人甲", "说话人乙", "说话人丙", "说话人丁", "说话人戊", "说话人己"];

function otherLabel(id: number): string {
  return OTHER_LABELS[id - 1] ?? `说话人${id}`;
}

export const REJECT_LABELS: Record<RejectReason, string> = {
  silence: "静音",
  noise: "环境噪声 / 喇叭",
  music: "音乐 / 电视",
  not_speech: "非语音",
  pitch_mismatch: "音高不符（不像你）",
  below_threshold: "相似度不足",
  no_enrollment: "尚未录入声纹",
};

type OtherCentroid = {
  id: number;
  vector: number[];
  pitchHz: number | null;
  count: number;
};

/**
 * Online speaker clustering with a dedicated "me" seat.
 * - meThreshold default 0.78
 * - otherThreshold default 0.7
 * - pitch ratio gate rejects "almost me" mismatches
 * - other people get separate 说话人甲/乙… seats
 */
export class OnlineSpeakerClusterer {
  meThreshold: number;
  otherThreshold: number;
  pitchRatioMin: number;
  pitchRatioMax: number;
  maxOthers: number;

  private meCentroid: number[] | null = null;
  private mePitchHz: number | null = null;
  private meCount = 0;
  private others: OtherCentroid[] = [];
  private nextOtherId = 1;

  constructor(options: ClustererOptions = {}) {
    this.meThreshold = options.meThreshold ?? 0.78;
    this.otherThreshold = options.otherThreshold ?? 0.7;
    this.pitchRatioMin = options.pitchRatioMin ?? 0.78;
    this.pitchRatioMax = options.pitchRatioMax ?? 1.28;
    this.maxOthers = options.maxOthers ?? 6;
  }

  enrollMe(embeddings: VoiceEmbedding[]): void {
    const vectors = embeddings.map((e) => e.vector).filter((v) => v.length > 0);
    if (vectors.length === 0) return;
    this.meCentroid = l2Normalize(averageVectors(vectors));
    const pitches = embeddings.map((e) => e.pitchHz).filter((p): p is number => p != null);
    this.mePitchHz = pitches.length ? pitches.reduce((a, b) => a + b, 0) / pitches.length : null;
    this.meCount = vectors.length;
  }

  clearOthers(): void {
    this.others = [];
    this.nextOtherId = 1;
  }

  reset(): void {
    this.meCentroid = null;
    this.mePitchHz = null;
    this.meCount = 0;
    this.clearOthers();
  }

  get hasEnrollment(): boolean {
    return this.meCentroid != null;
  }

  get peopleEstimate(): number {
    return (this.meCentroid ? 1 : 0) + this.others.length;
  }

  getOtherSeats(): SpeakerSeat[] {
    return this.others.map((o) => ({
      kind: "other" as const,
      id: o.id,
      label: otherLabel(o.id),
    }));
  }

  private pitchOk(pitchHz: number | null): boolean {
    if (this.mePitchHz == null || pitchHz == null) return true;
    const ratio = pitchHz / this.mePitchHz;
    return ratio >= this.pitchRatioMin && ratio <= this.pitchRatioMax;
  }

  assign(
    embedding: VoiceEmbedding,
    gateLabel: "silence" | "noise" | "music" | "speech" | "not_speech" = "speech",
  ): ClusterDecision {
    if (gateLabel === "silence") {
      return this.reject("silence");
    }
    if (gateLabel === "noise") {
      return this.reject("noise");
    }
    if (gateLabel === "music") {
      return this.reject("music");
    }
    if (gateLabel !== "speech") {
      return this.reject("not_speech");
    }

    if (!this.meCentroid) {
      return this.reject("no_enrollment");
    }

    const meSim = embeddingSimilarity(embedding.vector, this.meCentroid);

    if (meSim >= this.meThreshold) {
      if (!this.pitchOk(embedding.pitchHz)) {
        // High embedding sim but pitch off → do not treat as 你
        return this.assignOther(embedding, meSim, "pitch_mismatch");
      }
      this.updateMe(embedding);
      return {
        seat: { kind: "me", label: "你" },
        similarity: meSim,
        rejectReason: null,
        rejectLabel: null,
        peopleEstimate: this.peopleEstimate,
      };
    }

    // Near-me but below threshold with bad pitch → explicit reject reason for UI list
    if (meSim >= this.meThreshold - 0.08 && !this.pitchOk(embedding.pitchHz)) {
      return this.assignOther(embedding, meSim, "pitch_mismatch");
    }

    return this.assignOther(embedding, meSim, meSim < this.otherThreshold ? "below_threshold" : null);
  }

  private assignOther(
    embedding: VoiceEmbedding,
    meSim: number,
    preferReject: RejectReason | null,
  ): ClusterDecision {
    let bestIdx = -1;
    let bestSim = -1;
    for (let i = 0; i < this.others.length; i++) {
      const sim = embeddingSimilarity(embedding.vector, this.others[i]!.vector);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = i;
      }
    }

    if (bestIdx >= 0 && bestSim >= this.otherThreshold) {
      this.updateOther(bestIdx, embedding);
      const seat = this.others[bestIdx]!;
      return {
        seat: { kind: "other", id: seat.id, label: otherLabel(seat.id) },
        similarity: bestSim,
        rejectReason: preferReject === "pitch_mismatch" ? "pitch_mismatch" : null,
        rejectLabel: preferReject === "pitch_mismatch" ? REJECT_LABELS.pitch_mismatch : null,
        peopleEstimate: this.peopleEstimate,
      };
    }

    if (this.others.length < this.maxOthers) {
      const id = this.nextOtherId++;
      this.others.push({
        id,
        vector: embedding.vector.slice(),
        pitchHz: embedding.pitchHz,
        count: 1,
      });
      return {
        seat: { kind: "other", id, label: otherLabel(id) },
        similarity: Math.max(bestSim, 0),
        rejectReason: preferReject,
        rejectLabel: preferReject ? REJECT_LABELS[preferReject] : null,
        peopleEstimate: this.peopleEstimate,
      };
    }

    // Cap reached — attach to nearest or reject
    if (bestIdx >= 0) {
      this.updateOther(bestIdx, embedding);
      const seat = this.others[bestIdx]!;
      return {
        seat: { kind: "other", id: seat.id, label: otherLabel(seat.id) },
        similarity: bestSim,
        rejectReason: preferReject ?? "below_threshold",
        rejectLabel: REJECT_LABELS[preferReject ?? "below_threshold"],
        peopleEstimate: this.peopleEstimate,
      };
    }

    return {
      seat: null,
      similarity: meSim,
      rejectReason: preferReject ?? "below_threshold",
      rejectLabel: REJECT_LABELS[preferReject ?? "below_threshold"],
      peopleEstimate: this.peopleEstimate,
    };
  }

  private updateMe(embedding: VoiceEmbedding): void {
    if (!this.meCentroid) return;
    const n = this.meCount;
    const merged = this.meCentroid.map((v, i) => (v * n + (embedding.vector[i] ?? 0)) / (n + 1));
    this.meCentroid = l2Normalize(merged);
    this.meCount = n + 1;
    if (embedding.pitchHz != null) {
      this.mePitchHz =
        this.mePitchHz == null
          ? embedding.pitchHz
          : (this.mePitchHz * n + embedding.pitchHz) / (n + 1);
    }
  }

  private updateOther(index: number, embedding: VoiceEmbedding): void {
    const o = this.others[index]!;
    const n = o.count;
    const merged = o.vector.map((v, i) => (v * n + (embedding.vector[i] ?? 0)) / (n + 1));
    o.vector = l2Normalize(merged);
    o.count = n + 1;
    if (embedding.pitchHz != null) {
      o.pitchHz =
        o.pitchHz == null ? embedding.pitchHz : (o.pitchHz * n + embedding.pitchHz) / (n + 1);
    }
  }

  private reject(reason: RejectReason): ClusterDecision {
    return {
      seat: null,
      similarity: 0,
      rejectReason: reason,
      rejectLabel: REJECT_LABELS[reason],
      peopleEstimate: this.peopleEstimate,
    };
  }

  setThresholds(me: number, other: number): void {
    this.meThreshold = clamp(me, 0.5, 0.99);
    this.otherThreshold = clamp(other, 0.4, 0.95);
  }
}
