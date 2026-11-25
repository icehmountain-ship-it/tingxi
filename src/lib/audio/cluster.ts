import { embeddingSimilarity, type VoiceEmbedding } from "./features";
import { averageVectors, clamp, l2Normalize } from "./math";

export type VoiceTraits = {
  sex: "男" | "女" | "未知";
  ageBand: "大人" | "孩子" | "老人" | "未知";
};

export type SpeakerSeat =
  | { kind: "me"; label: "你"; traits?: VoiceTraits }
  | { kind: "other"; id: number; label: string; traits: VoiceTraits };

export type RejectReason =
  | "silence"
  | "noise"
  | "music"
  | "not_speech"
  | "pitch_mismatch"
  | "below_threshold"
  | "no_enrollment"
  | "near_me_hold";

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
  /** Soft-me floor: meSim above this must not seed a new other seat. */
  meProtectFloor?: number;
  /** Other must beat meSim by this margin to steal the frame. */
  otherWinMargin?: number;
  pitchRatioMin?: number;
  pitchRatioMax?: number;
  maxOthers?: number;
};

const OTHER_LABELS = ["说话人甲", "说话人乙", "说话人丙", "说话人丁", "说话人戊", "说话人己"];

function otherBaseLabel(id: number): string {
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
  near_me_hold: "接近本人，暂不分给别人",
};

type OtherCentroid = {
  id: number;
  vector: number[];
  pitchHz: number | null;
  count: number;
  traits: VoiceTraits;
};

/**
 * Pitch → rough sex / age band. Heuristic only; unknown is allowed.
 * Child / adult / elder boundaries are coarse and must not override seating.
 */
export function inferVoiceTraits(pitchHz: number | null): VoiceTraits {
  if (pitchHz == null || !Number.isFinite(pitchHz) || pitchHz <= 0) {
    return { sex: "未知", ageBand: "未知" };
  }
  if (pitchHz >= 255) {
    return { sex: "未知", ageBand: "孩子" };
  }
  if (pitchHz >= 175) {
    return { sex: "女", ageBand: "大人" };
  }
  if (pitchHz >= 85) {
    // Low adult male; very low may be elder-ish but keep adult unless extreme
    if (pitchHz < 110) {
      return { sex: "男", ageBand: "老人" };
    }
    return { sex: "男", ageBand: "大人" };
  }
  return { sex: "未知", ageBand: "未知" };
}

function formatOtherLabel(id: number, traits: VoiceTraits): string {
  const base = otherBaseLabel(id);
  const bits = [traits.sex, traits.ageBand].filter((x) => x !== "未知");
  return bits.length ? `${base}·${bits.join("·")}` : base;
}

function pitchCompatible(
  a: number | null,
  b: number | null,
  minRatio = 0.72,
  maxRatio = 1.38,
): boolean {
  if (a == null || b == null || a <= 0 || b <= 0) return true;
  const ratio = b / a;
  return ratio >= minRatio && ratio <= maxRatio;
}

/**
 * Online speaker clustering with a dedicated "me" seat.
 *
 * Lever for the two deadlocks (AQ soft-me → 甲; multi-person merge/split):
 * 1) Soft / falsetto "me" must stay "你": high meSim is not demoted by pitch alone;
 *    a protect floor stops near-me frames from seeding 说话人甲.
 * 2) Other seats need clear win over me + pitch compatibility so A/B don't merge
 *    across large pitch gaps; seats carry sex/age traits for basic readout.
 */
export class OnlineSpeakerClusterer {
  meThreshold: number;
  otherThreshold: number;
  meProtectFloor: number;
  otherWinMargin: number;
  pitchRatioMin: number;
  pitchRatioMax: number;
  maxOthers: number;

  private meCentroid: number[] | null = null;
  private mePitchHz: number | null = null;
  private meCount = 0;
  private others: OtherCentroid[] = [];
  private nextOtherId = 1;
  private lastWasMe = false;

  constructor(options: ClustererOptions = {}) {
    this.meThreshold = options.meThreshold ?? 0.78;
    this.otherThreshold = options.otherThreshold ?? 0.7;
    this.meProtectFloor =
      options.meProtectFloor ?? clamp(this.meThreshold - 0.12, 0.5, this.meThreshold);
    this.otherWinMargin = options.otherWinMargin ?? 0.05;
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
    this.lastWasMe = false;
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
      label: formatOtherLabel(o.id, o.traits),
      traits: o.traits,
    }));
  }

  private acceptMe(embedding: VoiceEmbedding, meSim: number): ClusterDecision {
    this.updateMe(embedding);
    this.lastWasMe = true;
    return {
      seat: {
        kind: "me",
        label: "你",
        traits: inferVoiceTraits(embedding.pitchHz ?? this.mePitchHz),
      },
      similarity: meSim,
      rejectReason: null,
      rejectLabel: null,
      peopleEstimate: this.peopleEstimate,
    };
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
    const { bestIdx, bestSim } = this.bestOtherMatch(embedding);

    // Strong me: trust embedding. Soft/falsetto must stay 「你」— do not demote on pitch alone.
    if (meSim >= this.meThreshold) {
      return this.acceptMe(embedding, meSim);
    }

    // Soft-me protect band: near enrollment, do not seed 说话人甲 as "other me".
    const protectFloor = this.lastWasMe
      ? Math.max(this.meProtectFloor - 0.03, 0.5)
      : this.meProtectFloor;

    if (meSim >= protectFloor) {
      const otherClearlyWins =
        bestIdx >= 0 &&
        bestSim >= this.otherThreshold &&
        bestSim >= meSim + this.otherWinMargin &&
        pitchCompatible(
          embedding.pitchHz,
          this.others[bestIdx]!.pitchHz,
          0.78,
          1.28,
        );

      if (otherClearlyWins) {
        return this.commitOther(bestIdx, embedding, bestSim, null);
      }

      // Hold as 你 rather than invent a new other seat from soft/near-me speech.
      return this.acceptMe(embedding, meSim);
    }

    return this.assignOther(embedding, meSim, bestIdx, bestSim);
  }

  private bestOtherMatch(embedding: VoiceEmbedding): { bestIdx: number; bestSim: number } {
    let bestIdx = -1;
    let bestSim = -1;
    for (let i = 0; i < this.others.length; i++) {
      const o = this.others[i]!;
      if (!pitchCompatible(embedding.pitchHz, o.pitchHz)) {
        continue;
      }
      const sim = embeddingSimilarity(embedding.vector, o.vector);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = i;
      }
    }
    return { bestIdx, bestSim };
  }

  private assignOther(
    embedding: VoiceEmbedding,
    meSim: number,
    bestIdx: number,
    bestSim: number,
  ): ClusterDecision {
    if (
      bestIdx >= 0 &&
      bestSim >= this.otherThreshold &&
      bestSim >= meSim + this.otherWinMargin
    ) {
      return this.commitOther(bestIdx, embedding, bestSim, null);
    }

    // Never open a new other seat while still near me (belt-and-suspenders).
    if (meSim >= this.meProtectFloor) {
      return this.acceptMe(embedding, meSim);
    }

    if (this.others.length < this.maxOthers) {
      const id = this.nextOtherId++;
      const traits = inferVoiceTraits(embedding.pitchHz);
      this.others.push({
        id,
        vector: embedding.vector.slice(),
        pitchHz: embedding.pitchHz,
        count: 1,
        traits,
      });
      this.lastWasMe = false;
      return {
        seat: { kind: "other", id, label: formatOtherLabel(id, traits), traits },
        similarity: Math.max(bestSim, 0),
        rejectReason: meSim < this.otherThreshold ? "below_threshold" : null,
        rejectLabel: meSim < this.otherThreshold ? REJECT_LABELS.below_threshold : null,
        peopleEstimate: this.peopleEstimate,
      };
    }

    if (bestIdx >= 0) {
      return this.commitOther(bestIdx, embedding, bestSim, "below_threshold");
    }

    this.lastWasMe = false;
    return {
      seat: null,
      similarity: meSim,
      rejectReason: "below_threshold",
      rejectLabel: REJECT_LABELS.below_threshold,
      peopleEstimate: this.peopleEstimate,
    };
  }

  private commitOther(
    index: number,
    embedding: VoiceEmbedding,
    sim: number,
    preferReject: RejectReason | null,
  ): ClusterDecision {
    this.updateOther(index, embedding);
    const seat = this.others[index]!;
    this.lastWasMe = false;
    return {
      seat: {
        kind: "other",
        id: seat.id,
        label: formatOtherLabel(seat.id, seat.traits),
        traits: seat.traits,
      },
      similarity: sim,
      rejectReason: preferReject,
      rejectLabel: preferReject ? REJECT_LABELS[preferReject] : null,
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
      o.traits = inferVoiceTraits(o.pitchHz);
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
    this.meProtectFloor = clamp(this.meThreshold - 0.12, 0.5, this.meThreshold);
  }
}
