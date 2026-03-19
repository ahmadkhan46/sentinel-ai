"use client";

import { motion } from "framer-motion";

export function HealthBar({ value }: { value: number | null }) {
  const pct = value !== null ? Math.round(value * 100) : 0;
  const color =
    pct > 70 ? "bg-emerald-500" : pct > 40 ? "bg-amber-400" : "bg-red-500";

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-slate-400 mb-1.5">
        <span>Health</span>
        <span className="font-medium text-slate-600">{value !== null ? `${pct}%` : "—"}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
        <motion.div
          className={`h-full rounded-full ${color}`}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}
