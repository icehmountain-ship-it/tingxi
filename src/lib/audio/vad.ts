import { FRAME_SIZE, HOP_SIZE, SAMPLE_RATE, analyzeFrame } from "./features";
import { rms } from "./math";

export type VadSegment = {
  start: number;
  end: number;
  samples: Float32Array;
};

export type VadOptions = {
  sampleRate?: number;
  energyThreshold?: number;
  minSpeechMs?: number;
  hangoverMs?: number;
};

/**
 * Lightweight energy + spectral VAD for carving speech-ish regions
 * before embedding / clustering.
 */
export function detectSpeechSegments(
  samples: ArrayLike<number>,
  options: VadOptions = {},
): VadSegment[] {
  const sampleRate = options.sampleRate ?? SAMPLE_RATE;
  const energyThreshold = options.energyThreshold ?? 0.015;
  const minSpeechSamples = Math.floor(((options.minSpeechMs ?? 180) / 1000) * sampleRate);
  const hangoverFrames = Math.max(
    1,
    Math.floor((((options.hangoverMs ?? 200) / 1000) * sampleRate) / HOP_SIZE),
  );

  const active: boolean[] = [];
  let prevPower: Float64Array | null = null;

  for (let start = 0; start + FRAME_SIZE <= samples.length; start += HOP_SIZE) {
    const frame = new Float64Array(FRAME_SIZE);
    for (let i = 0; i < FRAME_SIZE; i++) frame[i] = samples[start + i]!;
    const energy = rms(frame);
    const spec = analyzeFrame(frame, sampleRate, prevPower);
    prevPower = spec.power;
    const voiced =
      energy >= energyThreshold &&
      spec.flatness < 0.55 &&
      spec.zcr < 0.3 &&
      spec.energyDb > -45;
    active.push(voiced);
  }

  // Hangover smoothing
  const smoothed = active.slice();
  let hang = 0;
  for (let i = 0; i < smoothed.length; i++) {
    if (active[i]) {
      hang = hangoverFrames;
      smoothed[i] = true;
    } else if (hang > 0) {
      smoothed[i] = true;
      hang--;
    }
  }

  const segments: VadSegment[] = [];
  let segStart = -1;
  for (let i = 0; i <= smoothed.length; i++) {
    const on = i < smoothed.length ? smoothed[i]! : false;
    if (on && segStart < 0) segStart = i;
    if (!on && segStart >= 0) {
      const startSample = segStart * HOP_SIZE;
      const endSample = Math.min(samples.length, i * HOP_SIZE + FRAME_SIZE);
      if (endSample - startSample >= minSpeechSamples) {
        const slice = new Float32Array(endSample - startSample);
        for (let j = 0; j < slice.length; j++) slice[j] = samples[startSample + j]!;
        segments.push({ start: startSample, end: endSample, samples: slice });
      }
      segStart = -1;
    }
  }

  return segments;
}

export function isFrameActive(frame: ArrayLike<number>, threshold = 0.015): boolean {
  return rms(frame) >= threshold;
}
