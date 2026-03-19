"use client";

import { AuthGuard } from "@/components/AuthGuard";
import { Nav } from "@/components/Nav";
import { Badge } from "@/components/ui/Badge";
import { PageWrapper } from "@/components/ui/PageWrapper";
import { Spinner } from "@/components/ui/Spinner";
import { api, AlertOut } from "@/lib/api";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertTriangle,
  CheckCircle2,
  Eye,
  Filter,
  TrendingDown,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";

type FilterType = "all" | "open" | "acknowledged" | "resolved";

const SEVERITY_STYLES: Record<string, string> = {
  critical: "border-l-4 border-red-500 bg-red-50/40",
  warning:  "border-l-4 border-amber-400 bg-amber-50/30",
  info:     "border-l-4 border-blue-400 bg-blue-50/20",
};

function AlertCard({
  alert,
  index,
  onUpdate,
}: {
  alert: AlertOut;
  index: number;
  onUpdate: (a: AlertOut) => void;
}) {
  const [busy, setBusy] = useState(false);

  async function acknowledge() {
    setBusy(true);
    try { onUpdate(await api.acknowledgeAlert(alert.id)); } catch {}
    setBusy(false);
  }
  async function resolve() {
    setBusy(true);
    try { onUpdate(await api.resolveAlert(alert.id)); } catch {}
    setBusy(false);
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -20 }}
      transition={{ delay: index * 0.04, duration: 0.3 }}
      className={`rounded-2xl bg-white shadow-sm ${SEVERITY_STYLES[alert.severity] ?? "border-l-4 border-slate-300"}`}
    >
      <div className="p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3 min-w-0">
            <div className={`mt-0.5 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-lg ${
              alert.severity === "critical"
                ? "bg-red-100"
                : alert.severity === "warning"
                ? "bg-amber-100"
                : "bg-blue-100"
            }`}>
              <AlertTriangle className={`h-3.5 w-3.5 ${
                alert.severity === "critical"
                  ? "text-red-600"
                  : alert.severity === "warning"
                  ? "text-amber-600"
                  : "text-blue-600"
              }`} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <p className="text-sm font-semibold text-slate-800">{alert.title}</p>
                <Badge label={alert.severity} />
                <Badge label={alert.status} />
              </div>
              <p className="mt-1 text-xs text-slate-500 leading-relaxed line-clamp-2">
                {alert.message}
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-shrink-0 items-center gap-2">
            {alert.status === "open" && (
              <button
                onClick={acknowledge}
                disabled={busy}
                title="Acknowledge"
                className="flex items-center gap-1.5 rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-700 transition-colors hover:bg-amber-100 disabled:opacity-50"
              >
                <Eye className="h-3.5 w-3.5" />
                Ack
              </button>
            )}
            {alert.status !== "resolved" && (
              <button
                onClick={resolve}
                disabled={busy}
                title="Resolve"
                className="flex items-center gap-1.5 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-medium text-emerald-700 transition-colors hover:bg-emerald-100 disabled:opacity-50"
              >
                <CheckCircle2 className="h-3.5 w-3.5" />
                Resolve
              </button>
            )}
          </div>
        </div>

        {/* Meta */}
        <div className="mt-4 flex flex-wrap items-center gap-4 border-t border-slate-100 pt-3">
          {alert.rul_at_alert !== null && (
            <div className="flex items-center gap-1.5 text-xs text-slate-500">
              <TrendingDown className="h-3 w-3" />
              <span>RUL at alert:</span>
              <span className="font-semibold text-slate-700">
                {alert.rul_at_alert.toFixed(0)} cycles
              </span>
            </div>
          )}
          {alert.anomaly_score_at_alert !== null && (
            <div className="flex items-center gap-1.5 text-xs text-slate-500">
              <span>Score:</span>
              <span className="font-mono font-semibold text-slate-700">
                {alert.anomaly_score_at_alert.toFixed(3)}
              </span>
            </div>
          )}
          <Link
            href={`/assets/${alert.asset_id}`}
            className="flex items-center gap-1 text-xs font-medium text-violet-600 hover:underline"
          >
            View asset
          </Link>
          <span className="ml-auto text-xs text-slate-400">
            {formatDistanceToNow(new Date(alert.created_at), { addSuffix: true })}
          </span>
        </div>
      </div>
    </motion.div>
  );
}

function AlertsContent() {
  const [alerts, setAlerts] = useState<AlertOut[]>([]);
  const [filter, setFilter] = useState<FilterType>("open");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const params = filter === "all" ? {} : { status: filter };
    api.listAlerts(params)
      .then(setAlerts)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [filter]);

  function updateAlert(updated: AlertOut) {
    setAlerts((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));
  }

  const tabs: { key: FilterType; label: string }[] = [
    { key: "all",          label: "All" },
    { key: "open",         label: "Open" },
    { key: "acknowledged", label: "Acknowledged" },
    { key: "resolved",     label: "Resolved" },
  ];

  const criticalCount = alerts.filter((a) => a.severity === "critical" && a.status === "open").length;

  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto p-8">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">Monitoring</p>
          <h1 className="text-2xl font-bold text-slate-900">Alerts</h1>
        </div>
        {criticalCount > 0 && (
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            className="flex items-center gap-2 rounded-xl border border-red-200 bg-red-50 px-4 py-2"
          >
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500" />
            </span>
            <span className="text-sm font-semibold text-red-700">
              {criticalCount} critical alert{criticalCount !== 1 ? "s" : ""} require attention
            </span>
          </motion.div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 rounded-xl border border-slate-200 bg-white p-1 w-fit shadow-sm">
        <Filter className="ml-2 h-3.5 w-3.5 text-slate-400" />
        {tabs.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`relative rounded-lg px-4 py-1.5 text-sm font-medium transition-colors ${
              filter === key
                ? "text-slate-900"
                : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {filter === key && (
              <motion.div
                layoutId="tab-bg"
                className="absolute inset-0 rounded-lg bg-slate-100"
                transition={{ type: "spring", bounce: 0.2, duration: 0.35 }}
              />
            )}
            <span className="relative">{label}</span>
          </button>
        ))}
      </div>

      {/* Alert list */}
      {loading ? (
        <div className="flex h-40 items-center justify-center">
          <Spinner size={8} />
        </div>
      ) : (
        <AnimatePresence mode="popLayout">
          <div className="flex flex-col gap-3">
            {alerts.map((a, i) => (
              <AlertCard key={a.id} alert={a} index={i} onUpdate={updateAlert} />
            ))}
            {alerts.length === 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex h-40 items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white"
              >
                <div className="text-center">
                  <CheckCircle2 className="mx-auto h-8 w-8 text-emerald-400" />
                  <p className="mt-2 text-sm font-medium text-slate-600">All clear</p>
                  <p className="text-xs text-slate-400">No alerts in this category</p>
                </div>
              </motion.div>
            )}
          </div>
        </AnimatePresence>
      )}
    </div>
  );
}

export default function AlertsPage() {
  return (
    <AuthGuard>
      <div className="flex h-screen overflow-hidden bg-slate-50">
        <Nav />
        <main className="flex-1 overflow-hidden">
          <PageWrapper>
            <AlertsContent />
          </PageWrapper>
        </main>
      </div>
    </AuthGuard>
  );
}
