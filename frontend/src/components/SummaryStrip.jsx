function formatPct(x) {
  return `${Math.round(x * 1000) / 10}%`;
}

function formatMoney(x) {
  return `$${x.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export default function SummaryStrip({ summary }) {
  const cards = [
    {
      label: "Customers scored",
      value: summary.total_customers.toLocaleString(),
      tone: "neutral",
    },
    {
      label: "Predicted to churn",
      value: summary.predicted_churn_count.toLocaleString(),
      sub: formatPct(summary.churn_rate),
      tone: "high",
    },
    {
      label: "Average risk",
      value: formatPct(summary.average_risk),
      tone: "medium",
    },
    {
      label: "Monthly revenue at risk",
      value: formatMoney(summary.estimated_monthly_revenue_at_risk),
      sub: "from likely-to-churn accounts",
      tone: "high",
    },
  ];

  return (
    <section className="summary-strip">
      {cards.map((c) => (
        <div className={`summary-card summary-card--${c.tone}`} key={c.label}>
          <span className="summary-card__label">{c.label}</span>
          <span className="summary-card__value">{c.value}</span>
          {c.sub && <span className="summary-card__sub">{c.sub}</span>}
        </div>
      ))}
    </section>
  );
}
