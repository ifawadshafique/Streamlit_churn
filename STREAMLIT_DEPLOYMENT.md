# Streamlit deployment

This project now has a root Streamlit entry point:

```bash
streamlit run streamlit_app.py
```

The app reuses the existing trained model and pipeline:

- `models/churn_model.joblib`
- `models/scaler.joblib`
- `models/feature_names.joblib`
- `models/model_metadata.json`
- `src/preprocessing.py`
- `src/feature_engineering.py`
- `src/explain.py`

## Render

Create a Render Web Service from this repository.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

The included `render.yaml` contains the same settings.

## Streamlit Community Cloud

Create an app from the GitHub repository and set the main file to:

```text
streamlit_app.py
```

No separate FastAPI service or Vercel frontend is required for the Streamlit version.
