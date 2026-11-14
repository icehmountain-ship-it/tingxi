import { SAMPLE_RATE } from "./features";

export type CaptureHandlers = {
  onFrame?: (samples: Float32Array, sampleRate: number) => void;
  onError?: (error: Error) => void;
  onStateChange?: (state: CaptureState) => void;
};

export type CaptureState = "idle" | "requesting" | "running" | "stopped" | "error";

/**
 * Browser mic capture → mono Float32 frames at ~16 kHz via AudioWorklet / ScriptProcessor fallback.
 */
export class AudioCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private processor: ScriptProcessorNode | null = null;
  private state: CaptureState = "idle";
  private handlers: CaptureHandlers;

  constructor(handlers: CaptureHandlers = {}) {
    this.handlers = handlers;
  }

  getState(): CaptureState {
    return this.state;
  }

  private setState(state: CaptureState): void {
    this.state = state;
    this.handlers.onStateChange?.(state);
  }

  async start(): Promise<void> {
    if (this.state === "running" || this.state === "requesting") return;
    this.setState("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      this.stream = stream;

      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new Ctx({ sampleRate: SAMPLE_RATE });
      if (this.ctx.state === "suspended") await this.ctx.resume();

      this.source = this.ctx.createMediaStreamSource(stream);
      // ScriptProcessor is deprecated but widely available without extra worklet files.
      const bufferSize = 4096;
      this.processor = this.ctx.createScriptProcessor(bufferSize, 1, 1);
      this.processor.onaudioprocess = (ev) => {
        const input = ev.inputBuffer.getChannelData(0);
        const copy = new Float32Array(input.length);
        copy.set(input);
        const resampled = downsampleToRate(copy, this.ctx!.sampleRate, SAMPLE_RATE);
        this.handlers.onFrame?.(resampled, SAMPLE_RATE);
      };

      this.source.connect(this.processor);
      this.processor.connect(this.ctx.destination);
      this.setState("running");
    } catch (err) {
      this.setState("error");
      this.handlers.onError?.(err instanceof Error ? err : new Error(String(err)));
      await this.stop();
      throw err;
    }
  }

  async stop(): Promise<void> {
    try {
      this.processor?.disconnect();
      this.source?.disconnect();
      this.processor = null;
      this.source = null;
      this.stream?.getTracks().forEach((t) => t.stop());
      this.stream = null;
      if (this.ctx) {
        await this.ctx.close();
        this.ctx = null;
      }
    } finally {
      this.setState("stopped");
    }
  }
}

/** Simple decimation / zero-order hold resample toward 16 kHz. */
export function downsampleToRate(
  input: Float32Array,
  fromRate: number,
  toRate: number,
): Float32Array {
  if (fromRate === toRate) return input;
  if (fromRate < toRate) {
    // Upsample rarely needed; linear interpolate
    const ratio = fromRate / toRate;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const src = i * ratio;
      const i0 = Math.floor(src);
      const i1 = Math.min(i0 + 1, input.length - 1);
      const frac = src - i0;
      out[i] = input[i0]! * (1 - frac) + input[i1]! * frac;
    }
    return out;
  }
  const ratio = fromRate / toRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.floor((i + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let j = start; j < end && j < input.length; j++) {
      sum += input[j]!;
      count++;
    }
    out[i] = count ? sum / count : 0;
  }
  return out;
}

/** Record a fixed-duration clip from the mic (for enrollment). */
export async function recordClip(durationMs: number): Promise<Float32Array> {
  const chunks: Float32Array[] = [];
  let sampleRate = SAMPLE_RATE;
  const capture = new AudioCapture({
    onFrame: (samples, rate) => {
      sampleRate = rate;
      chunks.push(samples);
    },
  });
  await capture.start();
  await new Promise((r) => setTimeout(r, durationMs));
  await capture.stop();

  const total = chunks.reduce((n, c) => n + c.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  // Ensure nominal rate metadata path stays 16k
  return downsampleToRate(out, sampleRate, SAMPLE_RATE);
}
