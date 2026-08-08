import { useCallback, useRef, useState } from "react";

const ACCEPTED = ".csv,.xlsx,.xls";

export default function UploadPanel({ onFile, isLoading, error, fileName, apiOnline }) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback(
    (fileList) => {
      const file = fileList?.[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  return (
    <section className="upload-hero">
      <div className="upload-hero__intro">
        <span className="eyebrow">
          <span className="eyebrow__dot" aria-hidden="true" />
          Churn intelligence
        </span>
        <h1>
          Read the signal
          <br />
          before it drops.
        </h1>
        <p>
          Upload a customer file. Every row gets scored for churn risk, explained
          feature-by-feature, and rolled up into a report you can act on.
        </p>
        <div className={`api-status ${apiOnline ? "is-online" : "is-offline"}`}>
          <span className="api-status__pulse" />
          {apiOnline ? "Model service connected" : "Model service unreachable — start the API on :8000"}
        </div>
      </div>

      <div
        className={`dropzone ${isDragging ? "is-dragging" : ""} ${isLoading ? "is-loading" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />

        {isLoading ? (
          <>
            <div className="dropzone__spinner" aria-hidden="true" />
            <p className="dropzone__title">Scoring {fileName || "your file"}…</p>
            <p className="dropzone__hint">Cleaning rows, engineering features, running the model.</p>
          </>
        ) : (
          <>
            <svg className="dropzone__icon" viewBox="0 0 48 48" fill="none" aria-hidden="true">
              <path
                d="M24 6v24m0-24 8 8m-8-8-8 8"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M8 32v4a6 6 0 0 0 6 6h20a6 6 0 0 0 6-6v-4"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <p className="dropzone__title">Drop a CSV or Excel file, or click to browse</p>
            <p className="dropzone__hint">
              Telco-style columns (tenure, Contract, MonthlyCharges, …) — a header row is required.
            </p>
          </>
        )}
      </div>

      {error && (
        <p className="upload-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
