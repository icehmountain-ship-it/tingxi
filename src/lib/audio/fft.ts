/**
 * Lightweight radix-2 Cooley–Tukey FFT + helpers for spectral features.
 * No external ML / WASM model downloads.
 */

export type ComplexArray = { re: Float64Array; im: Float64Array };

export function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export function createComplex(size: number): ComplexArray {
  return { re: new Float64Array(size), im: new Float64Array(size) };
}

/** In-place radix-2 FFT. size must be power of 2. */
export function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n === 0 || (n & (n - 1)) !== 0) {
    throw new Error(`FFT size must be power of 2, got ${n}`);
  }

  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]!;
      re[i] = re[j]!;
      re[j] = tr;
      const ti = im[i]!;
      im[i] = im[j]!;
      im[j] = ti;
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang);
    const wlenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1;
      let wIm = 0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const uRe = re[i + k]!;
        const uIm = im[i + k]!;
        const vRe = re[i + k + half]! * wRe - im[i + k + half]! * wIm;
        const vIm = re[i + k + half]! * wIm + im[i + k + half]! * wRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + half] = uRe - vRe;
        im[i + k + half] = uIm - vIm;
        const nextWRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
      }
    }
  }
}

export function fftReal(signal: ArrayLike<number>, fftSize?: number): ComplexArray {
  const size = fftSize ?? nextPow2(signal.length);
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  const copy = Math.min(signal.length, size);
  for (let i = 0; i < copy; i++) re[i] = signal[i]!;
  fftInPlace(re, im);
  return { re, im };
}

export function magnitudeSpectrum(re: Float64Array, im: Float64Array): Float64Array {
  const half = re.length >> 1;
  const mag = new Float64Array(half + 1);
  for (let i = 0; i <= half; i++) {
    const r = re[i]!;
    const m = im[i]!;
    mag[i] = Math.hypot(r, m);
  }
  return mag;
}

export function powerSpectrum(re: Float64Array, im: Float64Array): Float64Array {
  const half = re.length >> 1;
  const power = new Float64Array(half + 1);
  for (let i = 0; i <= half; i++) {
    const r = re[i]!;
    const m = im[i]!;
    power[i] = r * r + m * m;
  }
  return power;
}

export function hannWindow(size: number): Float64Array {
  const w = new Float64Array(size);
  if (size === 1) {
    w[0] = 1;
    return w;
  }
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return w;
}

export function applyWindow(signal: ArrayLike<number>, window: ArrayLike<number>): Float64Array {
  const n = Math.min(signal.length, window.length);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = signal[i]! * window[i]!;
  return out;
}
