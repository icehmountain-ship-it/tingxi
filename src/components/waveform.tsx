"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

type WaveformProps = {
  level: number;
  active?: boolean;
  className?: string;
};

/** Lightweight canvas waveform driven by live RMS level. */
export function Waveform({ level, active = false, className }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef<number[]>(Array.from({ length: 64 }, () => 0));

  useEffect(() => {
    const hist = historyRef.current;
    hist.push(Math.min(1, level * 4));
    if (hist.length > 64) hist.shift();

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, active ? "#1f6f5b" : "#5c6b66");
    grad.addColorStop(1, active ? "#c4a35a" : "#8a9390");
    ctx.fillStyle = grad;

    const barW = w / hist.length;
    for (let i = 0; i < hist.length; i++) {
      const v = hist[i]!;
      const barH = Math.max(2, v * h * 0.9);
      const x = i * barW;
      const y = (h - barH) / 2;
      ctx.fillRect(x + 1, y, Math.max(1, barW - 2), barH);
    }
  }, [level, active]);

  return (
    <canvas
      ref={canvasRef}
      className={cn("h-24 w-full rounded-sm", className)}
      aria-hidden
    />
  );
}
