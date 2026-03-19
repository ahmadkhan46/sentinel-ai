import { clsx } from "clsx";

const variants: Record<string, string> = {
  operational: "bg-emerald-500/10 text-emerald-600 ring-1 ring-emerald-500/20",
  warning: "bg-amber-500/10 text-amber-600 ring-1 ring-amber-500/20",
  critical: "bg-red-500/10 text-red-600 ring-1 ring-red-500/20",
  offline: "bg-slate-500/10 text-slate-500 ring-1 ring-slate-500/20",
  open: "bg-red-500/10 text-red-600 ring-1 ring-red-500/20",
  acknowledged: "bg-amber-500/10 text-amber-600 ring-1 ring-amber-500/20",
  resolved: "bg-emerald-500/10 text-emerald-600 ring-1 ring-emerald-500/20",
  info: "bg-blue-500/10 text-blue-600 ring-1 ring-blue-500/20",
};

export function Badge({ label }: { label: string }) {
  const cls = variants[label] ?? "bg-slate-100 text-slate-500 ring-1 ring-slate-200";
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium capitalize",
        cls
      )}
    >
      {(label === "critical" || label === "open") && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-red-500" />
        </span>
      )}
      {label}
    </span>
  );
}
