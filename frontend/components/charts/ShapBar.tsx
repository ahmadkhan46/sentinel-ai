"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function ShapBar({ values }: { values: Record<string, number> }) {
  const data = Object.entries(values)
    .map(([feature, value]) => ({ feature, value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 12);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ left: 72, right: 20, top: 4, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
        <XAxis
          type="number"
          tick={{ fontSize: 10, fill: "#94a3b8" }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fontSize: 10, fill: "#64748b" }}
          width={72}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip
          contentStyle={{
            fontSize: 12,
            borderRadius: 12,
            border: "1px solid #e2e8f0",
            boxShadow: "0 4px 24px -4px rgba(0,0,0,0.12)",
          }}
          formatter={(v) => [typeof v === "number" ? v.toFixed(4) : v, "SHAP"]}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={14}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.value >= 0 ? "#ef4444" : "#7c3aed"}
              opacity={0.85}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
