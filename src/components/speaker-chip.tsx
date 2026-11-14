"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type SpeakerChipProps = {
  label: string;
  kind?: "me" | "other" | "reject" | "gate";
  className?: string;
};

export function SpeakerChip({ label, kind = "other", className }: SpeakerChipProps) {
  const variant =
    kind === "me" ? "success" : kind === "reject" ? "warning" : kind === "gate" ? "secondary" : "outline";

  return (
    <Badge
      variant={variant}
      className={cn(
        "animate-in fade-in zoom-in-95 duration-300",
        kind === "me" && "tracking-wide",
        className,
      )}
    >
      {label}
    </Badge>
  );
}
