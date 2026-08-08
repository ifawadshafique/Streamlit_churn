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
import { useState } from "react";

const RISK_COLORS = { High: "var(--risk-high)", Medium: "var(--risk-medium)", Low: "var(--risk-low)" };

function ChartTooltip({ active, payload, label, suffix }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <span className="chart-tooltip__label">{label}</span>
      <span className="chart-tooltip__value">
        {payload[0].value}
        {suffix}
      </span>
    </div>
  );
}

function bucketColor(bucketLabel) {
  const low = parseInt(bucketLabel, 10);
  if (low >= 70) return "var(--risk-high)";
  if (low >= 30) return "var(--risk-medium)";
  return "var(--risk-low)";
}

export default function ReportCharts({ histogram, segmentBreakdown }) {
  const segmentKeys = Object.keys(segmentBreakdown || {});
  const [activeSegment, setActiveSegment] = useState(segmentKeys[0] || null);

  const segmentData = activeSegment
    ? [...segmentBreakdown[activeSegment]].sort((a, b) => b.avg_risk - a.avg_risk)
    : [];

  return (
    <section className="charts-row">
      <div className="panel chart-panel">
        <div className="panel__header">
          <h2>Risk distribution</h2>
          <span className="panel__hint">Customers grouped by predicted churn probability</span>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={histogram} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="var(--border-soft)" vertical={false} />
            <XAxis
              dataKey="bucket"
              tick={{ fill: "var(--text-faint)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              axisLine={{ stroke: "var(--border)" }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "var(--text-faint)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              axisLine={false}
              tickLine={false}
              allowDecimals={false}
            />
            <Tooltip content={<ChartTooltip suffix=" customers" />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
            <Bar dataKey="count" radius={[3, 3, 0, 0]}>
              {histogram.map((entry) => (
                <Cell key={entry.bucket} fill={bucketColor(entry.bucket)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="panel chart-panel">
        <div className="panel__header">
          <h2>Risk by segment</h2>
          <div className="segment-tabs" role="tablist">
            {segmentKeys.map((key) => (
              <button
                key={key}
                role="tab"
                aria-selected={key === activeSegment}
                className={`segment-tab ${key === activeSegment ? "is-active" : ""}`}
                onClick={() => setActiveSegment(key)}
              >
                {key}
              </button>
            ))}
          </div>
        </div>
        {segmentData.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={segmentData}
              layout="vertical"
              margin={{ top: 8, right: 24, left: 8, bottom: 0 }}
            >
              <CartesianGrid stroke="var(--border-soft)" horizontal={false} />
              <XAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
                tick={{ fill: "var(--text-faint)", fontSize: 11, fontFamily: "var(--font-mono)" }}
                axisLine={{ stroke: "var(--border)" }}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="segment"
                width={130}
                tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                content={<ChartTooltip suffix=" avg. risk" />}
                cursor={{ fill: "rgba(255,255,255,0.03)" }}
                formatter={(v) => `${Math.round(v * 100)}%`}
              />
              <Bar dataKey="avg_risk" radius={[0, 3, 3, 0]} fill="var(--signal)" barSize={16} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="panel__empty">No segment data available.</p>
        )}
      </div>
    </section>
  );
}
