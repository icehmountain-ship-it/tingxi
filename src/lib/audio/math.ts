/** Vector / signal helpers used across the audio pipeline. */

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function mean(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i]!;
  return sum / values.length;
}

export function variance(values: ArrayLike<number>, avg = mean(values)): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const d = values[i]! - avg;
    sum += d * d;
  }
  return sum / values.length;
}

export function stddev(values: ArrayLike<number>, avg = mean(values)): number {
  return Math.sqrt(variance(values, avg));
}

export function rms(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    sum += v * v;
  }
  return Math.sqrt(sum / values.length);
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom < 1e-12) return 0;
  return clamp(dot / denom, -1, 1);
}

export function l2Normalize(vec: number[]): number[] {
  let n = 0;
  for (const v of vec) n += v * v;
  const norm = Math.sqrt(n);
  if (norm < 1e-12) return vec.map(() => 0);
  return vec.map((v) => v / norm);
}

export function averageVectors(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dim = vectors[0]!.length;
  const out = new Array(dim).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) out[i]! += v[i] ?? 0;
  }
  for (let i = 0; i < dim; i++) out[i]! /= vectors.length;
  return out;
}

export function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

export function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

export function dbFromPower(power: number, floor = 1e-12): number {
  return 10 * Math.log10(Math.max(power, floor));
}
