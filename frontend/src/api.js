const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export class ApiError extends Error {}

/**
 * Upload a CSV/XLSX file of customers and get back the full churn report
 * (summary stats, histogram, segment breakdowns, per-customer predictions).
 */
export async function uploadChurnFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/predict/batch/file`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

/**
 * Ask the model to explain a single customer's prediction (SHAP top reasons).
 * `customer` should be the raw customer record as returned in the batch
 * report's `customers` array (extra fields like ChurnProbability are ignored
 * by the backend's request schema).
 */
export async function explainCustomer(customer) {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(customer),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.detail || `Could not explain this customer (${res.status})`);
  }
  return res.json();
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) return false;
    const body = await res.json();
    return Boolean(body.model_loaded);
  } catch {
    return false;
  }
}
