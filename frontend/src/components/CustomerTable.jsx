import { useMemo, useState } from "react";
import SignalMeter from "./SignalMeter.jsx";

const RISK_FILTERS = ["All", "High", "Medium", "Low"];
const PAGE_SIZE = 25;

export default function CustomerTable({ customers, onSelectCustomer }) {
  const [query, setQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState("All");
  const [sortKey, setSortKey] = useState("ChurnProbability");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(0);

  const filtered = useMemo(() => {
    let rows = customers;
    if (riskFilter !== "All") {
      rows = rows.filter((r) => r.RiskLevel === riskFilter);
    }
    if (query.trim()) {
      const q = query.trim().toLowerCase();
      rows = rows.filter((r) =>
        [r.customerID, r.Contract, r.InternetService, r.PaymentMethod]
          .filter(Boolean)
          .some((v) => String(v).toLowerCase().includes(q))
      );
    }
    const sorted = [...rows].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      return sortDir === "asc"
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return sorted;
  }, [customers, query, riskFilter, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageRows = filtered.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  function toggleSort(key) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
    setPage(0);
  }

  const columns = [
    { key: "customerID", label: "Customer" },
    { key: "Contract", label: "Contract" },
    { key: "InternetService", label: "Internet" },
    { key: "tenure", label: "Tenure (mo)" },
    { key: "MonthlyCharges", label: "Monthly $" },
    { key: "ChurnProbability", label: "Risk" },
  ];

  return (
    <section className="panel table-panel">
      <div className="panel__header table-panel__header">
        <h2>Customers</h2>
        <div className="table-controls">
          <input
            className="table-search"
            type="search"
            placeholder="Search customer, contract, plan…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(0);
            }}
          />
          <div className="risk-filter" role="group" aria-label="Filter by risk level">
            {RISK_FILTERS.map((r) => (
              <button
                key={r}
                className={`risk-filter__btn ${riskFilter === r ? "is-active" : ""}`}
                onClick={() => {
                  setRiskFilter(r);
                  setPage(0);
                }}
              >
                {r}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col.key}>
                  <button className="th-sort" onClick={() => toggleSort(col.key)}>
                    {col.label}
                    {sortKey === col.key && (
                      <span className="th-sort__arrow">{sortDir === "asc" ? "↑" : "↓"}</span>
                    )}
                  </button>
                </th>
              ))}
              <th aria-hidden="true" />
            </tr>
          </thead>
          <tbody>
            {pageRows.map((c) => (
              <tr key={c.customerID} onClick={() => onSelectCustomer(c)} className="table-row">
                <td className="mono">{c.customerID}</td>
                <td>{c.Contract}</td>
                <td>{c.InternetService}</td>
                <td className="mono">{c.tenure}</td>
                <td className="mono">${Number(c.MonthlyCharges).toFixed(2)}</td>
                <td>
                  <SignalMeter probability={c.ChurnProbability} riskLevel={c.RiskLevel} />
                </td>
                <td className="table-row__chevron" aria-hidden="true">
                  ›
                </td>
              </tr>
            ))}
            {pageRows.length === 0 && (
              <tr>
                <td colSpan={columns.length + 1} className="table-empty">
                  No customers match this filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="table-pagination">
        <span>
          {filtered.length.toLocaleString()} customer{filtered.length === 1 ? "" : "s"}
        </span>
        <div className="table-pagination__controls">
          <button disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>
            Prev
          </button>
          <span className="mono">
            {page + 1} / {totalPages}
          </span>
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          >
            Next
          </button>
        </div>
      </div>
    </section>
  );
}
