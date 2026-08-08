const BAR_HEIGHTS = [5, 9, 13, 17, 21];

function riskColorVar(riskLevel) {
  if (riskLevel === "High") return "var(--risk-high)";
  if (riskLevel === "Medium") return "var(--risk-medium)";
  return "var(--risk-low)";
}

/**
 * Renders churn probability as a signal-strength meter (like a phone's signal
 * bars) rather than a plain progress bar — the higher the risk, the more the
 * "signal" has degraded from the customer's perspective. Filled bars = risk.
 */
export default function SignalMeter({ probability, riskLevel }) {
  const filledBars = Math.max(1, Math.round(probability * BAR_HEIGHTS.length));
  const color = riskColorVar(riskLevel);

  return (
    <span className="signal-meter" title={`${Math.round(probability * 100)}% churn risk`}>
      <span className="signal-meter__bars" aria-hidden="true">
        {BAR_HEIGHTS.map((h, i) => (
          <span
            key={h}
            className="signal-meter__bar"
            style={{
              height: `${h}px`,
              background: i < filledBars ? color : "var(--border)",
            }}
          />
        ))}
      </span>
      <span className="signal-meter__value" style={{ color }}>
        {Math.round(probability * 100)}%
      </span>
    </span>
  );
}
