"use client";

import { AuthGuard } from "@/components/AuthGuard";
import { Nav } from "@/components/Nav";
import { ShapBar } from "@/components/charts/ShapBar";
import { TrendChart } from "@/components/charts/TrendChart";
import { Badge } from "@/components/ui/Badge";
import { HealthBar } from "@/components/ui/HealthBar";
import { PageWrapper } from "@/components/ui/PageWrapper";
import { Spinner } from "@/components/ui/Spinner";
import { StatCard } from "@/components/ui/StatCard";
import { api, AssetOut, AssetTrend, InferenceOut } from "@/lib/api";
import { useOrgWebSocket } from "@/hooks/useWebSocket";
import { useAuth } from "@/lib/auth";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Brain,
  ChevronRight,
  Clock,
  Cpu,
  Play,
  RefreshCw,
  TrendingDown,
} from "lucide-react";
import Link from "next/link";
import { use, useEffect, useState } from "react";
import { formatDistanceToNow, format } from "date-fns";

function InferenceResultCard({ result }: { result: InferenceOut }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      className="mt-5 space-y-5"
    >
      {/* KPI row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard
          label="RUL"
          value={result.rul_prediction !== null ? `${result.rul_prediction.toFixed(0)}` : "—"}
          sub="cycles remaining"
          accent={result.rul_prediction !== null && result.rul_prediction < 30 ? "red" : "blue"}
          icon={<TrendingDown className="h-4 w-4" />}
        />
        <StatCard
          label="Health Index"
          value={result.health_index !== null ? `${Math.round(result.health_index * 100)}%` : "—"}
          accent={result.health_index !== null && result.health_index < 0.4 ? "red" : "green"}
          icon={<Activity className="h-4 w-4" />}
        />
        <StatCard
          label="Anomaly Score"
          value={result.anomaly_score !== null ? result.anomaly_score.toFixed(3) : "—"}
          accent={result.is_anomaly ? "red" : "default"}
          icon={<AlertTriangle className="h-4 w-4" />}
        />
        <StatCard
          label="Anomaly"
          value={result.is_anomaly ? "DETECTED" : "Normal"}
          accent={result.is_anomaly ? "red" : "green"}
          icon={<Brain className="h-4 w-4" />}
        />
      </div>

      {/* SHAP */}
      {result.shap_values && Object.keys(result.shap_values).length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="rounded-2xl border border-slate-200 bg-white p-5"
        >
          <div className="mb-4 flex items-center gap-2">
            <Brain className="h-4 w-4 text-violet-600" />
            <h3 className="text-sm font-semibold text-slate-800">SHAP Feature Attributions</h3>
            <span className="ml-auto text-xs text-slate-400">
              Red = increases anomaly risk · Purple = decreases
            </span>
          </div>
          <ShapBar values={result.shap_values} />
        </motion.div>
      )}
    </motion.div>
  );
}

function InferencePanel({ assetId, onResult }: { assetId: string; onResult: () => void }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<InferenceOut | null>(null);
  const [error, setError] = useState("");

  async function run() {
    setBusy(true);
    setError("");
    try {
      const r = await api.triggerInference(assetId);
      setResult(r);
      onResult();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Inference failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-600">
          <Cpu className="h-4 w-4 text-white" />
        </div>
        <div>
          <h2 className="text-sm font-semibold text-slate-800">Run Inference</h2>
          <p className="text-xs text-slate-400">Score latest sensor reading</p>
        </div>
        <button
          onClick={run}
          disabled={busy}
          className="ml-auto flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-md shadow-violet-500/20 transition-all hover:shadow-violet-500/40 disabled:opacity-60"
        >
          {busy ? (
            <>
              <RefreshCw className="h-3.5 w-3.5 animate-spin" />
              Scoring…
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5" />
              Score Now
            </>
          )}
        </button>
      </div>

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-600"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {result && <InferenceResultCard key={result.id} result={result} />}
      </AnimatePresence>
    </div>
  );
}

function HistoryRow({ inf, index }: { inf: InferenceOut; index: number }) {
  return (
    <motion.tr
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03 }}
      className="border-b border-slate-50 last:border-0 hover:bg-slate-50/80 transition-colors"
    >
      <td className="py-3 pl-5 pr-4 text-xs text-slate-500">
        <div className="flex items-center gap-1.5">
          <Clock className="h-3 w-3" />
          {format(new Date(inf.inferred_at), "HH:mm:ss")}
        </div>
      </td>
      <td className="py-3 px-4 text-xs text-slate-600">{inf.cycle ?? "—"}</td>
      <td className="py-3 px-4">
        {inf.is_anomaly !== null
          ? <Badge label={inf.is_anomaly ? "critical" : "operational"} />
          : <span className="text-xs text-slate-400">—</span>}
      </td>
      <td className="py-3 px-4 font-mono text-xs text-slate-700">
        {inf.anomaly_score?.toFixed(4) ?? "—"}
      </td>
      <td className="py-3 px-4 text-xs font-semibold text-slate-700">
        {inf.rul_prediction?.toFixed(0) ?? "—"}
      </td>
      <td className="py-3 pr-5 pl-4 text-xs text-slate-700">
        {inf.health_index !== null ? `${Math.round(inf.health_index * 100)}%` : "—"}
      </td>
    </motion.tr>
  );
}

function AssetDetailContent({ assetId }: { assetId: string }) {
  const { user } = useAuth();
  const [asset, setAsset] = useState<AssetOut | null>(null);
  const [trend, setTrend] = useState<AssetTrend | null>(null);
  const [history, setHistory] = useState<InferenceOut[]>([]);
  const [loading, setLoading] = useState(true);

  const wsEvent = useOrgWebSocket(user?.org_id);

  useEffect(() => {
    if (!wsEvent || wsEvent.asset_id !== assetId) return;
    api.getAsset(assetId).then(setAsset).catch(() => {});
    api.assetTrend(assetId).then(setTrend).catch(() => {});
    api.listInference(assetId).then(setHistory).catch(() => {});
  }, [wsEvent, assetId]);

  const fetchAll = () =>
    Promise.all([api.getAsset(assetId), api.assetTrend(assetId), api.listInference(assetId)])
      .then(([a, t, h]) => { setAsset(a); setTrend(t); setHistory(h); })
      .catch(() => {})
      .finally(() => setLoading(false));

  useEffect(() => { fetchAll(); }, [assetId]);

  if (loading)
    return <div className="flex h-full items-center justify-center"><Spinner size={10} /></div>;

  if (!asset)
    return <div className="p-8 text-sm text-slate-400">Asset not found.</div>;

  const chartData = trend
    ? trend.timestamps.map((_, i) => ({
        label: `#${i + 1}`,
        rul: trend.rul_values[i],
        hi: trend.health_index_values[i],
        anomaly: trend.anomaly_scores[i],
      }))
    : [];

  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto p-8">
      {/* Back + title */}
      <div>
        <Link
          href="/assets"
          className="mb-3 inline-flex items-center gap-1.5 text-xs font-medium text-slate-400 hover:text-violet-600 transition-colors"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to Assets
        </Link>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900">{asset.name}</h1>
            <p className="mt-0.5 flex items-center gap-1.5 text-sm text-slate-400">
              {asset.location && <><span>{asset.location}</span><ChevronRight className="h-3 w-3" /></>}
              {asset.serial_number ?? "No serial"}
            </p>
          </div>
          <Badge label={asset.status} />
        </div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard
          label="Health Index"
          value={asset.health_index !== null ? `${Math.round(asset.health_index * 100)}%` : "—"}
          accent={asset.health_index !== null && asset.health_index < 0.4 ? "red" : asset.health_index !== null && asset.health_index < 0.7 ? "amber" : "green"}
          icon={<Activity className="h-4 w-4" />}
        />
        <StatCard
          label="RUL"
          value={asset.last_rul !== null ? `${asset.last_rul} cycles` : "—"}
          accent={asset.last_rul !== null && asset.last_rul < 30 ? "red" : asset.last_rul !== null && asset.last_rul < 80 ? "amber" : "default"}
          icon={<TrendingDown className="h-4 w-4" />}
        />
        <StatCard
          label="Model"
          value={asset.model_name ?? "—"}
          sub={asset.model_version ?? undefined}
          icon={<Brain className="h-4 w-4" />}
        />
        <StatCard
          label="Last Inference"
          value={asset.last_inference_at ? formatDistanceToNow(new Date(asset.last_inference_at), { addSuffix: true }) : "Never"}
          icon={<Clock className="h-4 w-4" />}
        />
      </div>

      {/* Health bar (full width) */}
      <div className="rounded-2xl border border-slate-200 bg-white p-5">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-800">Health Status</h2>
          <span className="text-xs text-slate-400">Based on latest inference</span>
        </div>
        <HealthBar value={asset.health_index} />
      </div>

      {/* Inference panel */}
      <InferencePanel assetId={assetId} onResult={fetchAll} />

      {/* Trend chart */}
      {chartData.length > 0 && (
        <div className="rounded-2xl border border-slate-200 bg-white p-5">
          <div className="mb-4 flex items-center gap-2">
            <Activity className="h-4 w-4 text-violet-600" />
            <h2 className="text-sm font-semibold text-slate-800">RUL &amp; Health Trend</h2>
            <span className="ml-auto text-xs text-slate-400">{chartData.length} data points</span>
          </div>
          <TrendChart data={chartData} />
        </div>
      )}

      {/* Inference history */}
      {history.length > 0 && (
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
          <div className="flex items-center gap-2 border-b border-slate-100 px-5 py-4">
            <Clock className="h-4 w-4 text-slate-400" />
            <h2 className="text-sm font-semibold text-slate-800">Inference History</h2>
            <span className="ml-auto rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">
              {history.length} results
            </span>
          </div>
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50">
                {["Time", "Cycle", "Anomaly", "Score", "RUL", "Health"].map((h) => (
                  <th
                    key={h}
                    className={`py-2.5 px-4 text-xs font-semibold uppercase tracking-wide text-slate-400 ${
                      h === "Time" ? "pl-5" : h === "Health" ? "pr-5" : ""
                    }`}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.slice(0, 15).map((inf, i) => (
                <HistoryRow key={inf.id} inf={inf} index={i} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function AssetDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  return (
    <AuthGuard>
      <div className="flex h-screen overflow-hidden bg-slate-50">
        <Nav />
        <main className="flex-1 overflow-hidden">
          <PageWrapper>
            <AssetDetailContent assetId={id} />
          </PageWrapper>
        </main>
      </div>
    </AuthGuard>
  );
}
