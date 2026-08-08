import { useEffect, useState } from "react";
import "./App.css";
import UploadPanel from "./components/UploadPanel.jsx";
import SummaryStrip from "./components/SummaryStrip.jsx";
import ReportCharts from "./components/ReportCharts.jsx";
import CustomerTable from "./components/CustomerTable.jsx";
import CustomerDrawer from "./components/CustomerDrawer.jsx";
import { checkHealth, uploadChurnFile, ApiError } from "./api.js";

export default function App() {
  const [report, setReport] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [apiOnline, setApiOnline] = useState(true);
  const [selectedCustomer, setSelectedCustomer] = useState(null);

  useEffect(() => {
    checkHealth().then(setApiOnline);
  }, []);

  async function handleFile(file) {
    setFileName(file.name);
    setError(null);
    setIsLoading(true);
    try {
      const data = await uploadChurnFile(file);
      setReport(data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong reading that file.");
      setReport(null);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar__brand">
          <span className="topbar__mark" aria-hidden="true" />
          Signal
        </div>
        {report && (
          <button
            className="topbar__reset"
            onClick={() => {
              setReport(null);
              setFileName(null);
              setError(null);
            }}
          >
            New upload
          </button>
        )}
      </header>

      <main>
        {!report && (
          <UploadPanel
            onFile={handleFile}
            isLoading={isLoading}
            error={error}
            fileName={fileName}
            apiOnline={apiOnline}
          />
        )}

        {report && (
          <div className="report">
            <div className="report__source">
              <span className="eyebrow">
                <span className="eyebrow__dot" aria-hidden="true" />
                Report for {fileName}
              </span>
            </div>
            <SummaryStrip summary={report.summary} />
            <ReportCharts histogram={report.histogram} segmentBreakdown={report.segment_breakdown} />
            <CustomerTable customers={report.customers} onSelectCustomer={setSelectedCustomer} />
          </div>
        )}
      </main>

      <CustomerDrawer customer={selectedCustomer} onClose={() => setSelectedCustomer(null)} />
    </div>
  );
}
