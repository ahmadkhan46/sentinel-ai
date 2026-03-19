"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { motion } from "framer-motion";
import { clsx } from "clsx";
import {
  ActivitySquare,
  AlertTriangle,
  LayoutDashboard,
  LogOut,
  Server,
  Zap,
} from "lucide-react";

const links = [
  { href: "/dashboard", label: "Fleet Overview", icon: LayoutDashboard },
  { href: "/assets",    label: "Assets",         icon: Server },
  { href: "/alerts",    label: "Alerts",          icon: AlertTriangle },
];

export function Nav() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  return (
    <aside className="relative flex h-screen w-60 flex-shrink-0 flex-col bg-slate-950 px-4 py-6">
      {/* Subtle gradient glow at top */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-violet-600/10 to-transparent" />

      {/* Logo */}
      <Link href="/dashboard" className="mb-8 flex items-center gap-2.5 px-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 shadow-lg shadow-violet-500/30">
          <ActivitySquare className="h-4.5 w-4.5 text-white" strokeWidth={2.5} />
        </div>
        <div>
          <span className="block text-sm font-bold tracking-tight text-white">
            SENTINEL
          </span>
          <span className="block text-[10px] font-medium text-slate-500 uppercase tracking-widest">
            Industrial AI
          </span>
        </div>
      </Link>

      {/* Nav links */}
      <nav className="flex-1 space-y-1">
        <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-widest text-slate-600">
          Navigation
        </p>
        {links.map(({ href, label, icon: Icon }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors duration-150",
                active
                  ? "bg-white/10 text-white"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              )}
            >
              {active && (
                <motion.div
                  layoutId="nav-pill"
                  className="absolute inset-0 rounded-xl bg-gradient-to-r from-violet-600/30 to-indigo-600/20"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
              )}
              <Icon
                className={clsx(
                  "relative h-4 w-4 flex-shrink-0",
                  active ? "text-violet-400" : "text-slate-500 group-hover:text-slate-300"
                )}
              />
              <span className="relative">{label}</span>
              {active && (
                <span className="relative ml-auto h-1.5 w-1.5 rounded-full bg-violet-400" />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Live indicator */}
      <div className="mb-4 flex items-center gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-3 py-2">
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
        </span>
        <span className="text-xs font-medium text-emerald-400">Live Monitoring</span>
        <Zap className="ml-auto h-3 w-3 text-emerald-500" />
      </div>

      {/* User */}
      {user && (
        <div className="border-t border-slate-800 pt-4">
          <div className="flex items-center gap-3 rounded-xl px-2 py-2">
            <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 text-xs font-bold text-white">
              {user.full_name.charAt(0)}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-medium text-slate-300">{user.full_name}</p>
              <p className="truncate text-[10px] text-slate-600">{user.role}</p>
            </div>
            <button
              onClick={logout}
              title="Sign out"
              className="text-slate-600 transition-colors hover:text-slate-300"
            >
              <LogOut className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      )}
    </aside>
  );
}
