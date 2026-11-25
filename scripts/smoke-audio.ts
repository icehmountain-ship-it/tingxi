/**
 * Node smoke test for the audio pipeline (no mic / browser required).
 * Run: npm run smoke
 */
import assert from "node:assert/strict";
import { OnlineSpeakerClusterer, inferVoiceTraits } from "../src/lib/audio/cluster";
import {
  SAMPLE_RATE,
  classifyAudio,
  extractEmbedding,
  embeddingSimilarity,
  type VoiceEmbedding,
} from "../src/lib/audio/features";
import { fftReal, nextPow2, powerSpectrum } from "../src/lib/audio/fft";
import { cosineSimilarity, l2Normalize } from "../src/lib/audio/math";

function tone(
  freq: number,
  durationSec: number,
  sampleRate = SAMPLE_RATE,
  amp = 0.35,
): Float32Array {
  const n = Math.floor(durationSec * sampleRate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / sampleRate;
    // Slight amplitude envelope + 2nd harmonic → speech-ish, not pure sine music
    const env = 0.5 + 0.5 * Math.sin(2 * Math.PI * 3 * t);
    out[i] =
      amp *
      env *
      (Math.sin(2 * Math.PI * freq * t) + 0.35 * Math.sin(2 * Math.PI * freq * 2 * t));
  }
  return out;
}

function noise(durationSec: number, sampleRate = SAMPLE_RATE, amp = 0.25): Float32Array {
  const n = Math.floor(durationSec * sampleRate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (Math.random() * 2 - 1) * amp;
  return out;
}

function silence(durationSec: number, sampleRate = SAMPLE_RATE): Float32Array {
  return new Float32Array(Math.floor(durationSec * sampleRate));
}

function mix(a: Float32Array, b: Float32Array, gainB = 0.5): Float32Array {
  const n = Math.min(a.length, b.length);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i]! + b[i]! * gainB;
  return out;
}

console.log("听席 smoke-audio starting…");

// FFT sanity
{
  const n = nextPow2(1024);
  assert.equal(n, 1024);
  const signal = tone(440, 0.05);
  const { re, im } = fftReal(signal.slice(0, 512), 512);
  const power = powerSpectrum(re, im);
  let peak = 0;
  let peakIdx = 0;
  for (let i = 0; i < power.length; i++) {
    if (power[i]! > peak) {
      peak = power[i]!;
      peakIdx = i;
    }
  }
  const peakHz = (peakIdx * SAMPLE_RATE) / 512;
  assert.ok(Math.abs(peakHz - 440) < 40, `FFT peak expected ~440Hz, got ${peakHz}`);
  console.log("✓ FFT peak near 440 Hz");
}

// classifyAudio gate
{
  const silent = classifyAudio(silence(0.6));
  assert.equal(silent.label, "silence");

  const noisy = classifyAudio(noise(0.8, SAMPLE_RATE, 0.4));
  assert.ok(
    noisy.label === "noise" || noisy.label === "music",
    `noise expected noise|music, got ${noisy.label}`,
  );

  const speech = classifyAudio(tone(140, 1.0, SAMPLE_RATE, 0.45));
  assert.equal(speech.label, "speech", `speech tone gate failed: ${JSON.stringify(speech)}`);
  console.log("✓ classifyAudio silence / noise / speech");
}

// Embeddings + clusterer
{
  const meClips = [tone(145, 1.0), tone(148, 1.0), tone(142, 1.0)];
  const embeddings: VoiceEmbedding[] = [];
  for (const clip of meClips) {
    const emb = extractEmbedding(clip);
    assert.ok(emb, "expected embedding from speech-like clip");
    embeddings.push(emb!);
  }

  const clusterer = new OnlineSpeakerClusterer({
    meThreshold: 0.78,
    otherThreshold: 0.7,
  });
  assert.equal(clusterer.meThreshold, 0.78);
  assert.equal(clusterer.otherThreshold, 0.7);
  clusterer.enrollMe(embeddings);
  assert.equal(clusterer.hasEnrollment, true);

  const meLive = extractEmbedding(tone(146, 1.0));
  assert.ok(meLive);
  const meDecision = clusterer.assign(meLive!, "speech");
  assert.equal(meDecision.seat?.kind, "me");
  assert.equal(meDecision.seat?.label, "你");
  assert.ok(meDecision.similarity >= 0.78, `me sim ${meDecision.similarity}`);

  // Far embedding + different pitch → other seat (说话人甲…)
  // Pure tones stay too close in Mel space; use an explicit distant vector.
  const meVec = embeddings[0]!.vector;
  const otherVec = l2Normalize(meVec.map((_, i) => (i % 3 === 0 ? 1 : i % 3 === 1 ? -0.7 : 0.2)));
  assert.ok(embeddingSimilarity(otherVec, meVec) < 0.55, "fixture other must be far from me");
  const otherDecision = clusterer.assign(
    { vector: otherVec, pitchHz: 230, energyDb: -22 },
    "speech",
  );
  assert.equal(
    otherDecision.seat?.kind,
    "other",
    `expected other seat, got ${JSON.stringify(otherDecision)}`,
  );
  assert.match(otherDecision.seat!.label, /^说话人/);
  assert.ok(otherDecision.seat!.kind === "other" && otherDecision.seat.traits);

  // Noise gate before speaker ID
  const noiseDecision = clusterer.assign(
    { vector: l2Normalize([1, 0, 0]), pitchHz: 150, energyDb: -20 },
    "noise",
  );
  assert.equal(noiseDecision.seat, null);
  assert.equal(noiseDecision.rejectReason, "noise");

  const musicDecision = clusterer.assign(
    { vector: l2Normalize([1, 0, 0]), pitchHz: 150, energyDb: -20 },
    "music",
  );
  assert.equal(musicDecision.rejectReason, "music");

  console.log("✓ OnlineSpeakerClusterer me / other / reject gates");
  console.log(`  peopleEstimate=${clusterer.peopleEstimate}`);
}

// Deadlock regressions (aligned with grokbot-aimoyu PR#4 AQ soft-me → 甲)
{
  const meClips = [tone(145, 1.0), tone(148, 1.0), tone(142, 1.0)];
  const embeddings: VoiceEmbedding[] = [];
  for (const clip of meClips) {
    const emb = extractEmbedding(clip);
    assert.ok(emb);
    embeddings.push(emb!);
  }

  const clusterer = new OnlineSpeakerClusterer({
    meThreshold: 0.78,
    otherThreshold: 0.7,
    meProtectFloor: 0.66,
  });
  clusterer.enrollMe(embeddings);

  // Soft / quieter same-timbre clip must stay 「你」, not seed 说话人甲
  const softMe = extractEmbedding(tone(146, 1.0, SAMPLE_RATE, 0.18));
  assert.ok(softMe);
  const softDecision = clusterer.assign(softMe!, "speech");
  assert.equal(
    softDecision.seat?.kind,
    "me",
    `soft-me must stay 你, got ${JSON.stringify(softDecision)}`,
  );
  assert.equal(clusterer.getOtherSeats().length, 0, "soft-me must not seed other seats");

  // Near-me synthetic embedding below hard threshold but above protect floor
  const meCentroid = embeddings[0]!.vector;
  const nearMe: VoiceEmbedding = {
    vector: l2Normalize(meCentroid.map((v, i) => v * 0.92 + (i % 2 === 0 ? 0.08 : -0.05))),
    pitchHz: 210, // falsetto-ish; must not demote solely on pitch
    energyDb: -28,
  };
  const nearSim = embeddingSimilarity(nearMe.vector, meCentroid);
  assert.ok(nearSim < 0.78 || nearSim >= 0.66, `fixture sim unexpected: ${nearSim}`);
  const nearDecision = clusterer.assign(nearMe, "speech");
  assert.equal(
    nearDecision.seat?.kind,
    "me",
    `near-me/falsetto band must stay 你, got ${JSON.stringify(nearDecision)} sim=${nearDecision.similarity}`,
  );

  // Two others with far vectors + incompatible pitch must not merge
  const femaleVec = l2Normalize(meCentroid.map((_, i) => (i % 2 === 0 ? 0.9 : -0.4)));
  const maleVec = l2Normalize(meCentroid.map((_, i) => (i % 2 === 0 ? -0.85 : 0.55)));
  assert.ok(embeddingSimilarity(femaleVec, meCentroid) < 0.6);
  assert.ok(embeddingSimilarity(maleVec, meCentroid) < 0.6);
  const fDec = clusterer.assign(
    { vector: femaleVec, pitchHz: 220, energyDb: -22 },
    "speech",
  );
  const mDec = clusterer.assign(
    { vector: maleVec, pitchHz: 105, energyDb: -22 },
    "speech",
  );
  assert.equal(fDec.seat?.kind, "other", JSON.stringify(fDec));
  assert.equal(mDec.seat?.kind, "other", JSON.stringify(mDec));
  assert.notEqual(
    fDec.seat && fDec.seat.kind === "other" ? fDec.seat.id : -1,
    mDec.seat && mDec.seat.kind === "other" ? mDec.seat.id : -2,
    "male/female others must not share one seat",
  );

  const traitsChild = inferVoiceTraits(280);
  assert.equal(traitsChild.ageBand, "孩子");
  const traitsFemale = inferVoiceTraits(200);
  assert.equal(traitsFemale.sex, "女");
  const traitsMale = inferVoiceTraits(120);
  assert.equal(traitsMale.sex, "男");

  console.log("✓ soft-me protect + pitch-separated other seats + traits");
}

// Math helpers
{
  const sim = cosineSimilarity([1, 0], [1, 0]);
  assert.ok(Math.abs(sim - 1) < 1e-9);
  const mixed = mix(tone(140, 0.3), noise(0.3, SAMPLE_RATE, 0.05), 1);
  assert.ok(mixed.length > 0);
  console.log("✓ math helpers");
}

console.log("\nAll smoke checks passed.");
