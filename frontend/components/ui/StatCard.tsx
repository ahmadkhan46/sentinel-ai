"use client";

import { motion } from "framer-motion";

const accents: Record<string, { bg: string; text: string; ring: string }> = {
  green:   { bg: "from-emerald-500/10 to-emerald-500/5",  text: "text-emerald-600", ring: "ring-emerald-500/20" },
  amber:   { bg: "from-amber-500/10   to-amber-500/5",    text: "text-amber-600",   ring: "ring-amber-500/20"   },
  red:     { bg: "from-red-500/10     to-red-500/5",      text: "text-red-600",     ring: "ring-red-500/20"     },
  blue:    { bg: "from-violet-500/10  to-violet-500/5",   text: "text-violet-600",  ring: "ring-violet-500/20"  },
  default: { bg: "from-slate-100      to-slate-50",       text: "text-slate-800",   ring: "ring-slate-200"      },
};

export function StatCard({
  label,
  value,
  sub,
  icon,
  accent = "default",
}: {
  label: string;
  value: string | number;
  sub?: string;
  icon?: React.ReactNode;
  accent?: "green" | "amber" | "red" | "blue" | "default";
}) {
  const { bg, text, ring } = accents[accent];

  return (
    <motion.div
      whileHover={{ y: -2, boxShadow: "0 8px 30px -8px rgba(0,0,0,0.12)" }}
      transition={{ duration: 0.2 }}
      className={`relative overflow-hidden rounded-2xl bg-gradient-to-br ${bg} p-5 ring-1 ${ring}`}
    >
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</p>
          <p className={`mt-1.5 text-2xl font-bold ${text} truncate`}>{value}</p>
          {sub && <p className="mt-1 text-xs text-slate-400">{sub}</p>}
        </div>
        {icon && (
          <div className={`flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-white/80 ${text}`}>
            {icon}
          </div>
        )}
      </div>
    </motion.div>
  );
}
