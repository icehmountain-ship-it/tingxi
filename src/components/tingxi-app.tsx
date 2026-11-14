"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { SpeakerChip } from "@/components/speaker-chip";
import { Waveform } from "@/components/waveform";
import { recordClip } from "@/lib/audio/capture";
import { ENROLL_TARGET, TingxiEngine, type EngineSnapshot } from "@/lib/audio/engine";

const initial: EngineSnapshot = {
  status: "idle",
  enrolledCount: 0,
  meThreshold: 0.78,
  otherThreshold: 0.7,
  peopleEstimate: 0,
  lastGate: null,
  lastSeatLabel: null,
  lastRejectLabel: null,
  rejectHistory: [],
  level: 0,
  error: null,
};

export function TingxiApp() {
  const engineRef = useRef<TingxiEngine | null>(null);
  const [snap, setSnap] = useState<EngineSnapshot>(initial);
  const [message, setMessage] = useState("先录入三段你的说话声纹，再开始听席。");
  const [enrolling, setEnrolling] = useState(false);
  const [, startTransition] = useTransition();

  useEffect(() => {
    const engine = new TingxiEngine();
    engineRef.current = engine;
    return engine.subscribe((s) => {
      startTransition(() => setSnap(s));
    });
  }, []);

  const ready = snap.enrolledCount >= ENROLL_TARGET;
  const listening = snap.status === "listening";

  async function handleEnroll() {
    if (!engineRef.current || enrolling) return;
    setEnrolling(true);
    setMessage("正在录制约 2 秒…请对着麦克风自然说话。");
    try {
      const clip = await recordClip(2000);
      const result = engineRef.current.enrollFromSamples(clip);
      setMessage(result.message);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "无法访问麦克风");
    } finally {
      setEnrolling(false);
    }
  }

  async function toggleListen() {
    const engine = engineRef.current;
    if (!engine) return;
    if (listening) {
      await engine.stopListening();
      setMessage("已停止听席。");
      return;
    }
    try {
      await engine.startListening();
      setMessage("正在听席：识别「你」与其他人，并拒绝噪声/音乐。");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "无法开始听席");
    }
  }

  function clearAll() {
    engineRef.current?.clearEnrollment();
    setMessage("声纹已清除（仅保存在本机 localStorage）。");
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-10 px-6 py-10 md:py-14">
      <header className="space-y-3">
        <p className="text-sm tracking-[0.2em] text-muted-foreground">浏览器端声纹席位</p>
        <h1 className="font-display text-5xl font-semibold tracking-tight text-foreground md:text-6xl">
          听席
        </h1>
        <p className="max-w-xl text-base leading-relaxed text-muted-foreground">
          嘈杂场景里区分「你」与其他人。喇叭、音乐、电视与噪声会被拒绝；其他说话人分席为甲、乙…
          声纹只存在本机 localStorage。
        </p>
      </header>

      <section className="space-y-4" aria-label="声纹录入">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-display text-2xl">录入声纹</h2>
            <p className="text-sm text-muted-foreground">
              需要 {ENROLL_TARGET} 段短语音 · 已完成 {snap.enrolledCount}/{ENROLL_TARGET}
            </p>
          </div>
          <div className="flex gap-2">
            <Button onClick={handleEnroll} disabled={enrolling || listening}>
              {enrolling ? "录制中…" : ready ? "再录一段" : "录入一段"}
            </Button>
            <Button variant="outline" onClick={clearAll} disabled={listening}>
              清除
            </Button>
          </div>
        </div>
        <div className="flex gap-2">
          {Array.from({ length: ENROLL_TARGET }).map((_, i) => (
            <div
              key={i}
              className={`h-1.5 flex-1 rounded-full transition-colors duration-500 ${
                i < snap.enrolledCount ? "bg-primary" : "bg-secondary"
              }`}
            />
          ))}
        </div>
      </section>

      <section className="space-y-4" aria-label="实时听席">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-display text-2xl">实时听席</h2>
            <p className="text-sm text-muted-foreground">{message}</p>
          </div>
          <Button
            size="lg"
            variant={listening ? "destructive" : "default"}
            onClick={toggleListen}
            disabled={!ready && !listening}
          >
            {listening ? "停止" : "开始听席"}
          </Button>
        </div>

        <div className="overflow-hidden rounded-md border border-border/60 bg-waveform-panel px-3 py-4">
          <Waveform level={snap.level} active={listening} />
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm text-muted-foreground">当前席位</span>
          {snap.lastSeatLabel ? (
            <SpeakerChip
              label={snap.lastSeatLabel}
              kind={snap.lastSeatLabel === "你" ? "me" : "other"}
            />
          ) : (
            <SpeakerChip label="无人发言" kind="gate" />
          )}
          {snap.lastGate && <SpeakerChip label={`门控：${gateLabel(snap.lastGate)}`} kind="gate" />}
          <span className="ml-auto text-sm text-muted-foreground">
            人数估计 · {snap.peopleEstimate}
          </span>
        </div>
        {snap.error && <p className="text-sm text-destructive">{snap.error}</p>}
      </section>

      <section className="space-y-5" aria-label="灵敏度">
        <div>
          <h2 className="font-display text-2xl">灵敏度</h2>
          <p className="text-sm text-muted-foreground">
            「你」默认 0.78 · 其他人默认 0.70 · 另有音高比门限防止误认
          </p>
        </div>
        <label className="block space-y-2">
          <div className="flex justify-between text-sm">
            <span>识别为「你」的阈值</span>
            <span className="tabular-nums text-muted-foreground">
              {snap.meThreshold.toFixed(2)}
            </span>
          </div>
          <Slider
            min={0.55}
            max={0.95}
            step={0.01}
            value={[snap.meThreshold]}
            onValueChange={([v]) => engineRef.current?.setThresholds(v!, snap.otherThreshold)}
          />
        </label>
        <label className="block space-y-2">
          <div className="flex justify-between text-sm">
            <span>其他人分席阈值</span>
            <span className="tabular-nums text-muted-foreground">
              {snap.otherThreshold.toFixed(2)}
            </span>
          </div>
          <Slider
            min={0.45}
            max={0.9}
            step={0.01}
            value={[snap.otherThreshold]}
            onValueChange={([v]) => engineRef.current?.setThresholds(snap.meThreshold, v!)}
          />
        </label>
      </section>

      <section className="space-y-3" aria-label="不会当成你的">
        <h2 className="font-display text-2xl">不会当成你的</h2>
        <p className="text-sm text-muted-foreground">
          被门控或聚类拒绝的原因会出现在这里（静音、噪声、音乐、音高不符等）。
        </p>
        <ul className="flex flex-wrap gap-2">
          {snap.rejectHistory.length === 0 ? (
            <li>
              <SpeakerChip label="尚无拒绝记录" kind="gate" />
            </li>
          ) : (
            snap.rejectHistory.map((reason, i) => (
              <li key={`${reason}-${i}`}>
                <SpeakerChip label={reason} kind="reject" />
              </li>
            ))
          )}
        </ul>
      </section>
    </div>
  );
}

function gateLabel(gate: string): string {
  switch (gate) {
    case "silence":
      return "静音";
    case "noise":
      return "噪声";
    case "music":
      return "音乐";
    case "speech":
      return "语音";
    default:
      return gate;
  }
}
