# Signal — Churn Dashboard (React frontend)

Upload a CSV or Excel file of customers, get a full churn report: summary
stats, a risk distribution chart, risk-by-segment breakdowns, and a
searchable/sortable customer table. Click any customer to see the top
SHAP-based reasons behind their score.

This talks to the FastAPI backend in `../app/api.py` — start that first.

## 1. Start the backend

From the project root (one level up):

```bash
pip install -r requirements.txt
# if you haven't trained a model yet:
python src/generate_data.py && python src/preprocessing.py && \
  python src/feature_engineering.py && python src/train.py

uvicorn app.api:app --reload --port 8000
```

Confirm it's up: `curl http://localhost:8000/health`

## 2. Run the frontend

```bash
cd frontend
npm install
cp .env.example .env   # adjust VITE_API_BASE_URL if your API isn't on :8000
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`).

## What the file needs

Any CSV/XLSX with the Telco-style columns the model was trained on —
`tenure`, `Contract`, `MonthlyCharges`, `InternetService`, etc. (see
`data/telco_churn.csv` for a reference file, or `app/api.py`'s
`CustomerInput` model for the full field list). A `customerID` column is
optional — rows are auto-numbered if it's missing. An existing `Churn`
column, if present, is ignored (it's not needed to score new customers).

## Build for production

```bash
npm run build
```

Outputs static files to `dist/` — serve with any static host, pointing
`VITE_API_BASE_URL` at your deployed API.
