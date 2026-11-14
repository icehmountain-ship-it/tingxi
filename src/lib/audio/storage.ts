import type { VoiceEmbedding } from "./features";

const STORAGE_KEY = "tingxi.voiceprint.v1";

export type StoredVoiceprint = {
  version: 1;
  createdAt: string;
  updatedAt: string;
  enrollments: VoiceEmbedding[];
  meThreshold: number;
  otherThreshold: number;
};

function canUseStorage(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

/** Voiceprint lives in localStorage only — no server / auth / db. */
export function loadVoiceprint(): StoredVoiceprint | null {
  if (!canUseStorage()) return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as StoredVoiceprint;
    if (parsed?.version !== 1 || !Array.isArray(parsed.enrollments)) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function saveVoiceprint(data: {
  enrollments: VoiceEmbedding[];
  meThreshold: number;
  otherThreshold: number;
  createdAt?: string;
}): StoredVoiceprint {
  const now = new Date().toISOString();
  const record: StoredVoiceprint = {
    version: 1,
    createdAt: data.createdAt ?? now,
    updatedAt: now,
    enrollments: data.enrollments,
    meThreshold: data.meThreshold,
    otherThreshold: data.otherThreshold,
  };
  if (canUseStorage()) {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(record));
  }
  return record;
}

export function clearVoiceprint(): void {
  if (!canUseStorage()) return;
  window.localStorage.removeItem(STORAGE_KEY);
}

export function upsertEnrollment(
  embedding: VoiceEmbedding,
  thresholds: { meThreshold: number; otherThreshold: number },
  maxClips = 3,
): StoredVoiceprint {
  const existing = loadVoiceprint();
  const enrollments = [...(existing?.enrollments ?? []), embedding].slice(-maxClips);
  return saveVoiceprint({
    createdAt: existing?.createdAt,
    enrollments,
    meThreshold: thresholds.meThreshold,
    otherThreshold: thresholds.otherThreshold,
  });
}
