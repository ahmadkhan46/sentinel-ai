"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface DataPoint {
  label: string;
  rul: number | null;
  hi: number | null;
  anomaly: number | null;
}

export function TrendChart({ data }: { data: DataPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={data} margin={{ top: 5, right: 16, left: -10, bottom: 0 }}>
        <defs>
          <linearGradient id="grad-rul" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#7c3aed" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#7c3aed" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="grad-hi" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#10b981" stopOpacity={0.2} />
            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="grad-anom" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#f59e0b" stopOpacity={0.15} />
            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 10, fill: "#94a3b8" }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          yAxisId="left"
          tick={{ fontSize: 10, fill: "#94a3b8" }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          domain={[0, 1]}
          tick={{ fontSize: 10, fill: "#94a3b8" }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip
          contentStyle={{
            fontSize: 12,
            borderRadius: 12,
            border: "1px solid #e2e8f0",
            boxShadow: "0 4px 24px -4px rgba(0,0,0,0.12)",
            background: "white",
          }}
          formatter={(val, name) => [
            typeof val === "number" ? val.toFixed(2) : val,
            name,
          ]}
        />
        <Legend
          wrapperStyle={{ fontSize: 11, paddingTop: 12, color: "#64748b" }}
          iconType="circle"
          iconSize={6}
        />
        <Area
          yAxisId="left"
          type="monotone"
          dataKey="rul"
          name="RUL (cycles)"
          stroke="#7c3aed"
          strokeWidth={2}
          fill="url(#grad-rul)"
          dot={false}
          activeDot={{ r: 4, strokeWidth: 0 }}
        />
        <Area
          yAxisId="right"
          type="monotone"
          dataKey="hi"
          name="Health Index"
          stroke="#10b981"
          strokeWidth={2}
          fill="url(#grad-hi)"
          dot={false}
          activeDot={{ r: 4, strokeWidth: 0 }}
        />
        <Area
          yAxisId="right"
          type="monotone"
          dataKey="anomaly"
          name="Anomaly Score"
          stroke="#f59e0b"
          strokeWidth={1.5}
          strokeDasharray="5 3"
          fill="url(#grad-anom)"
          dot={false}
          activeDot={{ r: 4, strokeWidth: 0 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
