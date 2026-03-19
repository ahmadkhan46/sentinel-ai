"use client";

import { AuthGuard } from "@/components/AuthGuard";
import { Nav } from "@/components/Nav";
import { Badge } from "@/components/ui/Badge";
import { HealthBar } from "@/components/ui/HealthBar";
import { PageWrapper } from "@/components/ui/PageWrapper";
import { Spinner } from "@/components/ui/Spinner";
import { api, AssetOut } from "@/lib/api";
import { motion } from "framer-motion";
import { Activity, Server, TrendingDown } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";

function AssetRow({ asset, index }: { asset: AssetOut; index: number }) {
  const hi = asset.health_index;
  const hiPct = hi !== null ? Math.round(hi * 100) : null;

  return (
    <motion.tr
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.04, duration: 0.3 }}
      className="group border-b border-slate-100 last:border-0 hover:bg-slate-50/80 transition-colors"
    >
      <td className="py-4 pl-6 pr-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg bg-slate-900">
            <Server className="h-3.5 w-3.5 text-slate-300" />
          </div>
          <div>
            <Link
              href={`/assets/${asset.id}`}
              className="text-sm font-semibold text-slate-800 hover:text-violet-700 transition-colors"
            >
              {asset.name}
            </Link>
            <p className="text-xs text-slate-400">{asset.serial_number ?? "No serial"}</p>
          </div>
        </div>
      </td>
      <td className="py-4 px-4 text-sm text-slate-500">{asset.location ?? "—"}</td>
      <td className="py-4 px-4">
        <Badge label={asset.status} />
      </td>
      <td className="py-4 px-4" style={{ minWidth: 140 }}>
        <HealthBar value={hi} />
      </td>
      <td className="py-4 px-4">
        <div className="flex items-center gap-1.5">
          <TrendingDown className="h-3.5 w-3.5 text-slate-400" />
          <span className={`text-sm font-semibold ${
            (asset.last_rul ?? 999) < 30
              ? "text-red-600"
              : (asset.last_rul ?? 999) < 80
              ? "text-amber-600"
              : "text-slate-700"
          }`}>
            {asset.last_rul !== null ? `${asset.last_rul}` : "—"}
          </span>
          {asset.last_rul !== null && (
            <span className="text-xs text-slate-400">cycles</span>
          )}
        </div>
      </td>
      <td className="py-4 px-4">
        {asset.model_name ? (
          <div className="inline-flex items-center gap-1 rounded-md bg-violet-50 px-2 py-0.5 text-xs font-medium text-violet-700 ring-1 ring-violet-200">
            <Activity className="h-3 w-3" />
            {asset.model_name}
          </div>
        ) : (
          <span className="text-xs text-slate-400">—</span>
        )}
      </td>
      <td className="py-4 pr-6 pl-4 text-right text-xs text-slate-400">
        {asset.last_inference_at
          ? formatDistanceToNow(new Date(asset.last_inference_at), { addSuffix: true })
          : "Never"}
      </td>
    </motion.tr>
  );
}

function AssetsContent() {
  const [assets, setAssets] = useState<AssetOut[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.listAssets().then(setAssets).catch(() => {}).finally(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner size={10} />
      </div>
    );

  return (
    <div className="flex h-full flex-col gap-6 overflow-y-auto p-8">
      <div className="flex items-end justify-between">
        <div>
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wide">Fleet</p>
          <h1 className="text-2xl font-bold text-slate-900">Assets</h1>
        </div>
        <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-500 shadow-sm">
          {assets.length} assets
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-slate-100 bg-slate-50">
              {["Asset", "Location", "Status", "Health", "RUL", "Model", "Last Check"].map((h) => (
                <th
                  key={h}
                  className={`py-3 px-4 text-xs font-semibold uppercase tracking-wide text-slate-400 ${
                    h === "Asset" ? "pl-6" : h === "Last Check" ? "pr-6 text-right" : ""
                  }`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {assets.map((a, i) => (
              <AssetRow key={a.id} asset={a} index={i} />
            ))}
            {assets.length === 0 && (
              <tr>
                <td colSpan={7} className="py-16 text-center text-sm text-slate-400">
                  No assets found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function AssetsPage() {
  return (
    <AuthGuard>
      <div className="flex h-screen overflow-hidden bg-slate-50">
        <Nav />
        <main className="flex-1 overflow-hidden">
          <PageWrapper>
            <AssetsContent />
          </PageWrapper>
        </main>
      </div>
    </AuthGuard>
  );
}
