import { applyWindow, fftReal, hannWindow, powerSpectrum } from "./fft";
import {
  averageVectors,
  clamp,
  cosineSimilarity,
  dbFromPower,
  hzToMel,
  l2Normalize,
  mean,
  melToHz,
  rms,
  stddev,
} from "./math";

export const SAMPLE_RATE = 16000;
export const FRAME_SIZE = 512;
export const HOP_SIZE = 256;
export const MEL_BANDS = 24;
export const EMBEDDING_DIM = MEL_BANDS + 4; // mel + pitch stats

export type AudioClass = "silence" | "noise" | "music" | "speech";

export type ClassifyResult = {
  label: AudioClass;
  confidence: number;
  reasons: string[];
  energyDb: number;
  spectralCentroidHz: number;
  flatness: number;
  flux: number;
  zcr: number;
  pitchHz: number | null;
};

export type VoiceEmbedding = {
  vector: number[];
  pitchHz: number | null;
  energyDb: number;
};

function buildMelFilterbank(
  fftBins: number,
  sampleRate: number,
  nMels: number,
  fMin = 50,
  fMax = sampleRate / 2,
): Float64Array[] {
  const melMin = hzToMel(fMin);
  const melMax = hzToMel(fMax);
  const melPoints = new Array(nMels + 2);
  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = melMin + ((melMax - melMin) * i) / (nMels + 1);
  }
  const hzPoints = melPoints.map((m) => melToHz(m));
  const bins = hzPoints.map((hz) => Math.floor(((fftBins - 1) * hz) / (sampleRate / 2)));

  const filters: Float64Array[] = [];
  for (let m = 1; m <= nMels; m++) {
    const filter = new Float64Array(fftBins);
    const left = bins[m - 1]!;
    const center = bins[m]!;
    const right = bins[m + 1]!;
    for (let k = left; k < center; k++) {
      if (center !== left) filter[k] = (k - left) / (center - left);
    }
    for (let k = center; k < right; k++) {
      if (right !== center) filter[k] = (right - k) / (right - center);
    }
    filters.push(filter);
  }
  return filters;
}

const MEL_FILTERS = buildMelFilterbank(FRAME_SIZE / 2 + 1, SAMPLE_RATE, MEL_BANDS);

function zeroCrossingRate(frame: ArrayLike<number>): number {
  if (frame.length < 2) return 0;
  let crossings = 0;
  for (let i = 1; i < frame.length; i++) {
    if ((frame[i]! >= 0) !== (frame[i - 1]! >= 0)) crossings++;
  }
  return crossings / (frame.length - 1);
}

function spectralCentroid(power: Float64Array, sampleRate: number): number {
  let num = 0;
  let den = 0;
  const binHz = sampleRate / ((power.length - 1) * 2);
  for (let i = 0; i < power.length; i++) {
    const p = power[i]!;
    num += i * binHz * p;
    den += p;
  }
  return den > 1e-12 ? num / den : 0;
}

function spectralFlatness(power: Float64Array): number {
  let logSum = 0;
  let sum = 0;
  let count = 0;
  for (let i = 1; i < power.length; i++) {
    const p = Math.max(power[i]!, 1e-12);
    logSum += Math.log(p);
    sum += p;
    count++;
  }
  if (count === 0 || sum <= 0) return 1;
  const geo = Math.exp(logSum / count);
  const arith = sum / count;
  return clamp(geo / arith, 0, 1);
}

function spectralFlux(prev: Float64Array | null, curr: Float64Array): number {
  if (!prev) return 0;
  let sum = 0;
  const n = Math.min(prev.length, curr.length);
  for (let i = 0; i < n; i++) {
    const d = curr[i]! - prev[i]!;
    if (d > 0) sum += d;
  }
  return sum / n;
}

/** Autocorrelation pitch estimate in Hz, or null if unreliable. */
export function estimatePitch(
  frame: ArrayLike<number>,
  sampleRate = SAMPLE_RATE,
  minHz = 70,
  maxHz = 400,
): number | null {
  const energy = rms(frame);
  if (energy < 0.01) return null;

  const minLag = Math.floor(sampleRate / maxHz);
  const maxLag = Math.min(Math.floor(sampleRate / minHz), frame.length - 1);
  if (maxLag <= minLag) return null;

  let bestLag = -1;
  let bestCorr = 0;
  let r0 = 0;
  for (let i = 0; i < frame.length; i++) r0 += frame[i]! * frame[i]!;
  if (r0 < 1e-8) return null;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let corr = 0;
    for (let i = 0; i < frame.length - lag; i++) {
      corr += frame[i]! * frame[i + lag]!;
    }
    corr /= r0;
    if (corr > bestCorr) {
      bestCorr = corr;
      bestLag = lag;
    }
  }

  if (bestLag < 0 || bestCorr < 0.35) return null;
  return sampleRate / bestLag;
}

function melEnergies(power: Float64Array): number[] {
  return MEL_FILTERS.map((filter) => {
    let e = 0;
    for (let i = 0; i < filter.length; i++) e += filter[i]! * power[i]!;
    return dbFromPower(e);
  });
}

export type FrameSpectrum = {
  power: Float64Array;
  mel: number[];
  centroid: number;
  flatness: number;
  flux: number;
  zcr: number;
  pitchHz: number | null;
  energyDb: number;
};

const WINDOW = hannWindow(FRAME_SIZE);

export function analyzeFrame(
  samples: ArrayLike<number>,
  sampleRate = SAMPLE_RATE,
  prevPower: Float64Array | null = null,
): FrameSpectrum {
  const frame = samples.length >= FRAME_SIZE ? samples : padFrame(samples, FRAME_SIZE);
  const windowed = applyWindow(sliceFrame(frame, FRAME_SIZE), WINDOW);
  const { re, im } = fftReal(windowed, FRAME_SIZE);
  const power = powerSpectrum(re, im);
  const energy = rms(windowed);
  return {
    power,
    mel: melEnergies(power),
    centroid: spectralCentroid(power, sampleRate),
    flatness: spectralFlatness(power),
    flux: spectralFlux(prevPower, power),
    zcr: zeroCrossingRate(windowed),
    pitchHz: estimatePitch(windowed, sampleRate),
    energyDb: dbFromPower(energy * energy),
  };
}

function sliceFrame(samples: ArrayLike<number>, size: number): Float64Array {
  const out = new Float64Array(size);
  const n = Math.min(samples.length, size);
  for (let i = 0; i < n; i++) out[i] = samples[i]!;
  return out;
}

function padFrame(samples: ArrayLike<number>, size: number): Float64Array {
  const out = new Float64Array(size);
  for (let i = 0; i < samples.length && i < size; i++) out[i] = samples[i]!;
  return out;
}

/**
 * Gate before speaker ID: silence | noise | music | speech.
 * Rejects horns / TV / music / ambient as non-speech.
 */
export function classifyAudio(
  samples: ArrayLike<number>,
  sampleRate = SAMPLE_RATE,
): ClassifyResult {
  if (samples.length < FRAME_SIZE) {
    return {
      label: "silence",
      confidence: 1,
      reasons: ["片段过短"],
      energyDb: dbFromPower(rms(samples) ** 2),
      spectralCentroidHz: 0,
      flatness: 1,
      flux: 0,
      zcr: 0,
      pitchHz: null,
    };
  }

  const centroids: number[] = [];
  const flatnesses: number[] = [];
  const fluxes: number[] = [];
  const zcrs: number[] = [];
  const pitches: number[] = [];
  const energies: number[] = [];
  let prevPower: Float64Array | null = null;
  let speechLikeFrames = 0;
  let totalFrames = 0;

  for (let start = 0; start + FRAME_SIZE <= samples.length; start += HOP_SIZE) {
    const frame = new Float64Array(FRAME_SIZE);
    for (let i = 0; i < FRAME_SIZE; i++) frame[i] = samples[start + i]!;
    const spec = analyzeFrame(frame, sampleRate, prevPower);
    prevPower = spec.power;
    centroids.push(spec.centroid);
    flatnesses.push(spec.flatness);
    fluxes.push(spec.flux);
    zcrs.push(spec.zcr);
    energies.push(spec.energyDb);
    if (spec.pitchHz != null) pitches.push(spec.pitchHz);

    const voiced =
      spec.energyDb > -40 &&
      spec.flatness < 0.5 &&
      spec.zcr < 0.28 &&
      (spec.pitchHz != null || (spec.zcr > 0.015 && spec.flatness < 0.35));
    if (voiced) speechLikeFrames++;
    totalFrames++;
  }

  const energyDb = mean(energies);
  const spectralCentroidHz = mean(centroids);
  const flatness = mean(flatnesses);
  const flux = mean(fluxes);
  const zcr = mean(zcrs);
  const pitchHz = pitches.length ? mean(pitches) : null;
  const pitchFrameRatio = totalFrames ? pitches.length / totalFrames : 0;
  const speechRatio = totalFrames ? speechLikeFrames / totalFrames : 0;
  const reasons: string[] = [];

  if (energyDb < -45) {
    reasons.push("能量过低");
    return {
      label: "silence",
      confidence: 0.95,
      reasons,
      energyDb,
      spectralCentroidHz,
      flatness,
      flux,
      zcr,
      pitchHz,
    };
  }

  // Broadband noise / hiss
  if (flatness > 0.55 && energyDb > -35) {
    reasons.push("频谱平坦（噪声/气流）");
    return {
      label: "noise",
      confidence: clamp(flatness, 0.55, 0.99),
      reasons,
      energyDb,
      spectralCentroidHz,
      flatness,
      flux,
      zcr,
      pitchHz,
    };
  }

  if (zcr > 0.28 && spectralCentroidHz > 2500) {
    reasons.push("高频刺耳（喇叭/鸣笛）");
    return {
      label: "noise",
      confidence: 0.85,
      reasons,
      energyDb,
      spectralCentroidHz,
      flatness,
      flux,
      zcr,
      pitchHz,
    };
  }

  // Speech: stable pitch + low flatness wins over flux-based music heuristics
  const pitchStability =
    pitches.length >= 3 ? 1 - clamp(stddev(pitches) / (mean(pitches) || 1), 0, 1) : pitchFrameRatio;

  if (
    pitchHz != null &&
    pitchFrameRatio >= 0.2 &&
    flatness < 0.4 &&
    zcr < 0.25 &&
    spectralCentroidHz < 2200
  ) {
    reasons.push("检测到浊音基频");
    return {
      label: "speech",
      confidence: clamp(0.55 + pitchFrameRatio * 0.4, 0.55, 0.98),
      reasons,
      energyDb,
      spectralCentroidHz,
      flatness,
      flux,
      zcr,
      pitchHz,
    };
  }

  // Music / TV-like: weak pitch lock + elevated flatness/flux
  if (speechRatio < 0.25 && flux > 0.002 && flatness > 0.25 && pitchStability < 0.4) {
    reasons.push("节奏性频谱变化（音乐/电视）");
    return {
      label: "music",
      confidence: 0.8,
      reasons,
      energyDb,
      spectralCentroidHz,
      flatness,
      flux,
      zcr,
      pitchHz,
    };
  }

  if (speechRatio >= 0.2 && pitchHz != null && zcr < 0.25) {
    reasons.push("检测到浊音基频");
    return {
      label: "speech",
      confidence: clamp(0.55 + speechRatio * 0.4, 0.55, 0.98),
      reasons,
      energyDb,
      spectralCentroidHz,
      flatness,
      flux,
      zcr,
      pitchHz,
    };
  }

  if (speechRatio >= 0.15 && energyDb > -38) {
    reasons.push("类语音能量分布");
    return {
      label: "speech",
      confidence: 0.6,
      reasons,
      energyDb,
      spectralCentroidHz,
      flatness,
      flux,
      zcr,
      pitchHz,
    };
  }

  reasons.push("未通过语音门限");
  return {
    label: "noise",
    confidence: 0.7,
    reasons,
    energyDb,
    spectralCentroidHz,
    flatness,
    flux,
    zcr,
    pitchHz,
  };
}

/** Mel + pitch embedding via FFT (no heavy ML model). */
export function extractEmbedding(
  samples: ArrayLike<number>,
  sampleRate = SAMPLE_RATE,
): VoiceEmbedding | null {
  const classified = classifyAudio(samples, sampleRate);
  if (classified.label !== "speech") return null;

  const melAccum: number[][] = [];
  const pitches: number[] = [];
  const energies: number[] = [];
  let prevPower: Float64Array | null = null;

  for (let start = 0; start + FRAME_SIZE <= samples.length; start += HOP_SIZE) {
    const frame = new Float64Array(FRAME_SIZE);
    for (let i = 0; i < FRAME_SIZE; i++) frame[i] = samples[start + i]!;
    const spec = analyzeFrame(frame, sampleRate, prevPower);
    prevPower = spec.power;
    if (spec.energyDb < -42) continue;
    melAccum.push(spec.mel);
    energies.push(spec.energyDb);
    if (spec.pitchHz != null) pitches.push(spec.pitchHz);
  }

  if (melAccum.length < 2) return null;

  const melMean = averageVectors(melAccum);
  const pitchMean = pitches.length ? mean(pitches) : 0;
  const pitchStd = pitches.length ? stddev(pitches) : 0;
  const pitchPresence = pitches.length / Math.max(melAccum.length, 1);
  const energyMean = mean(energies);

  const raw = [
    ...melMean.map((v) => v / 60), // rough dB scale normalize
    pitchMean / 400,
    pitchStd / 100,
    pitchPresence,
    (energyMean + 60) / 60,
  ];

  return {
    vector: l2Normalize(raw),
    pitchHz: pitches.length ? pitchMean : null,
    energyDb: energyMean,
  };
}

export function embeddingSimilarity(a: number[], b: number[]): number {
  return cosineSimilarity(a, b);
}
