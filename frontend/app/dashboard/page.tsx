"use client";

import { AuthGuard } from "@/components/AuthGuard";
import { Nav } from "@/components/Nav";
import { Badge } from "@/components/ui/Badge";
import { HealthBar } from "@/components/ui/HealthBar";
import { PageWrapper } from "@/components/ui/PageWrapper";
import { Spinner } from "@/components/ui/Spinner";
import { StatCard } from "@/components/ui/StatCard";
import { useAuth } from "@/lib/auth";
import { api, AssetOut, FleetSummary } from "@/lib/api";
import { useOrgWebSocket } from "@/hooks/useWebSocket";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Server,
  TrendingDown,
  Wifi,
  XCircle,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";

function AssetCard({ asset, index }: { asset: AssetOut; index: number }) {
  const hi = asset.health_index;

  const statusIcon = {
    operational: <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />,
    warning: <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />,
    critical: <XCircle className="h-3.5 w-3.5 text-red-500" />,
    offline: <Wifi className="h-3.5 w-3.5 text-slate-400" />,
  }[asset.status] ?? null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.35, ease: "easeOut" }}
    >
      <Link href={`/assets/${asset.id}`}>
        <motion.div
          whileHover={{ y: -3, boxShadow: "0 16px 40px -8px rgba(0,0,0,0.12)" }}
          transition={{ duration: 0.2 }}
          className="group rounded-2xl border border-slate-200/80 bg-white p-5 cursor-pointer transition-colors"
        >
          {/* Header */}
          <div className="flex items-start justify-between gap-3 mb-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-slate-900 shadow-sm">
                <Server className="h-4 w-4 text-slate-300" />
              </div>
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-slate-800 group-hover:text-violet-700 transition-colors">
                  {asset.name}
                </p>
                <p className="text-xs text-slate-400 truncate">{asset.location ?? "—"}</p>
              </div>
            </div>
            <Badge label={asset.status} />
          </div>

          {/* Health bar */}
          <HealthBar value={hi} />

          {/* Bottom stats */}
          <div className="mt-4 flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-xs text-slate-500">
              <TrendingDown className="h-3.5 w-3.5" />
              <span>RUL</span>
              <span className={`font-semibold ${
                (asset.last_rul ?? 999) < 30
                  ? "text-red-600"
                  : (asset.last_rul ?? 999) < 80
                  ? "text-amber-600"
                  : "text-slate-700"
              }`}>
                {asset.last_rul !== null ? `${asset.last_rul} cycles` : "—"}
              </span>
            </div>
            {asset.last_inference_at && (
              <span className="text-[10px] text-slate-400">
                {formatDistanceToNow(new Date(asset.last_inference_at), { addSuffix: true })}
              </span>
            )}
          </div>

          {/* Model tag */}
          {asset.model_name && (
            <div className="mt-3 inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500">
              <Activity className="h-2.5 w-2.5" />
              {asset.model_name}
            </div>
          )}
        </motion.div>
      </Link>
    </motion.div>
  );
}

function LiveBanner({ event }: { event: { asset_id: string; rul: number | null; is_anomaly: boolean | null } }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -8 }}
      className="flex items-center gap-3 rounded-2xl border border-violet-200 bg-gradient-to-r from-violet-50 to-indigo-50 px-5 py-3"
    >
      <span className="relative flex h-2.5 w-2.5">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-75" />
        <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-violet-500" />
      </span>
      <span className="text-sm font-medium text-violet-700">
        Live inference — Asset <span className="font-mono">{event.asset_id.slice(-8)}</span>
        {" "}&middot; RUL {event.rul ?? "N/A"} cycles
        {event.is_anomaly && (
          <span className="ml-2 text-red-600 font-semibold">⚠ Anomaly</span>
        )}
      </span>
    </motion.div>
  );
}

function DashboardContent() {
  const { user } = useAuth();
  const [summary, setSummary] = useState<FleetSummary | null>(null);
  const [assets, setAssets] = useState<AssetOut[]>([]);
  const [loading, setLoading] = useState(true);

  const wsEvent = useOrgWebSocket(user?.org_id);

  useEffect(() => {
    if (!wsEvent) return;
    api.listAssets().then(setAssets).catch(() => {});
  }, [wsEvent]);

  useEffect(() => {
    Promise.all([api.fleetSummary(), api.listAssets()])
      .then(([s, a]) => { setSummary(s); setAssets(a); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner size={10} />
      </div>
    );

  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto p-8">
      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-medium text-slate-400">
            {greeting}, {user?.full_name?.split(" ")[0]}
          </p>
          <h1 className="text-2xl font-bold text-slate-900">Fleet Overview</h1>
        </div>
        <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-500 shadow-sm">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
          {assets.length} assets monitored
        </div>
      </div>

      {/* Live event banner */}
      <AnimatePresence mode="wait">
        {wsEvent && <LiveBanner key={wsEvent.asset_id} event={wsEvent} />}
      </AnimatePresence>

      {/* KPI Cards */}
      {summary && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-6">
          {[
            { label: "Total Assets",  value: summary.total_assets,  accent: "default" as const, icon: <Server className="h-4 w-4" /> },
            { label: "Operational",   value: summary.operational,   accent: "green"   as const, icon: <CheckCircle2 className="h-4 w-4" /> },
            { label: "Warning",       value: summary.warning,       accent: "amber"   as const, icon: <AlertTriangle className="h-4 w-4" /> },
            { label: "Critical",      value: summary.critical,      accent: "red"     as const, icon: <XCircle className="h-4 w-4" /> },
            { label: "Open Alerts",   value: summary.open_alerts,   accent: "red"     as const, icon: <AlertTriangle className="h-4 w-4" /> },
            {
              label: "Avg Health",
              value: summary.avg_health_index !== null
                ? `${Math.round(summary.avg_health_index * 100)}%` : "—",
              accent: "blue" as const,
              icon: <Activity className="h-4 w-4" />,
            },
          ].map((card, i) => (
            <motion.div
              key={card.label}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
            >
              <StatCard {...card} />
            </motion.div>
          ))}
        </div>
      )}

      {/* Section label */}
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-semibold text-slate-700">Assets</h2>
        <div className="flex-1 border-t border-slate-200" />
        <span className="text-xs text-slate-400">{assets.length} total</span>
      </div>

      {/* Asset grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {assets.map((a, i) => (
          <AssetCard key={a.id} asset={a} index={i} />
        ))}
        {assets.length === 0 && (
          <div className="col-span-3 flex h-40 items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white">
            <p className="text-sm text-slate-400">No assets found.</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <AuthGuard>
      <div className="flex h-screen overflow-hidden bg-slate-50">
        <Nav />
        <main className="flex-1 overflow-hidden">
          <PageWrapper>
            <DashboardContent />
          </PageWrapper>
        </main>
      </div>
    </AuthGuard>
  );
}
