import { useEffect, useState } from "react";
import SignalMeter from "./SignalMeter.jsx";
import { explainCustomer } from "../api.js";

function readableFeature(name) {
  // Turn one-hot / engineered feature names into something a human reads
  // naturally, e.g. "Contract_Month-to-month" -> "Contract: Month-to-month"
  return name.replace(/_/g, " → ").replace(/([a-z])([A-Z])/g, "$1 $2");
}

export default function CustomerDrawer({ customer, onClose }) {
  const [reasons, setReasons] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!customer) return;
    setReasons(null);
    setError(null);
    setLoading(true);

    const { ChurnProbability, Prediction, RiskLevel, TenureGroup, ...raw } = customer;
    explainCustomer(raw)
      .then((res) => setReasons(res.top_reasons))
      .catch((e) => setError(e.message || "Could not load explanation."))
      .finally(() => setLoading(false));
  }, [customer]);

  if (!customer) return null;

  return (
    <div className="drawer-overlay" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Customer detail">
        <header className="drawer__header">
          <div>
            <span className="drawer__eyebrow">Customer</span>
            <h2 className="mono">{customer.customerID}</h2>
          </div>
          <button className="drawer__close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <div className="drawer__risk">
          <SignalMeter probability={customer.ChurnProbability} riskLevel={customer.RiskLevel} />
          <span className={`badge badge--${customer.RiskLevel.toLowerCase()}`}>{customer.Prediction}</span>
        </div>

        <dl className="drawer__facts">
          <div>
            <dt>Contract</dt>
            <dd>{customer.Contract}</dd>
          </div>
          <div>
            <dt>Tenure</dt>
            <dd>{customer.tenure} mo</dd>
          </div>
          <div>
            <dt>Internet</dt>
            <dd>{customer.InternetService}</dd>
          </div>
          <div>
            <dt>Monthly charges</dt>
            <dd>${Number(customer.MonthlyCharges).toFixed(2)}</dd>
          </div>
          <div>
            <dt>Payment method</dt>
            <dd>{customer.PaymentMethod}</dd>
          </div>
          <div>
            <dt>Paperless billing</dt>
            <dd>{customer.PaperlessBilling}</dd>
          </div>
        </dl>

        <section className="drawer__reasons">
          <h3>Why this score</h3>
          {loading && <p className="drawer__hint">Computing SHAP contributions…</p>}
          {error && <p className="upload-error">{error}</p>}
          {reasons && (
            <ul className="reason-list">
              {reasons.map((r) => (
                <li key={r.feature} className={`reason-list__item is-${r.direction}`}>
                  <span className="reason-list__icon" aria-hidden="true">
                    {r.direction === "increases" ? "▲" : "▼"}
                  </span>
                  <div>
                    <span className="reason-list__feature">{readableFeature(r.feature)}</span>
                    <span className="reason-list__meta">
                      value: {String(r.value)} · impact {r.impact > 0 ? "+" : ""}
                      {r.impact}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </aside>
    </div>
  );
}
