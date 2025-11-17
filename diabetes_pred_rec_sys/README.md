# Diabetes Prediction Apps

This folder now contains two complementary applications powered by the same serialized mult-class diabetes model.

## Project layout

- `app.py` – Gradio UI for quick predictions.
- `dashboard_app.py` – Streamlit experience with a sleek interface, persistent database, and analytics dashboard.
- `ml_utils.py` – Shared feature-engineering + inference helpers.
- `models/` – Place the trained model artifact here (expected filename `best_model_random_forest.pkl`).
- `data/` – SQLite database (`predictions.db`) is created here when using the dashboard.

## Setup

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the trained model artifact to `diabetes_pred_rec_sys/models/best_model_random_forest.pkl`.

## Run the Gradio predictor

```bash
python app.py
```

## Run the Streamlit dashboard

```bash
streamlit run dashboard_app.py
```

### Dashboard features

- Rich input form with validation and immediate feedback.
- Each submission is stored in `data/predictions.db` (SQLite) along with model metadata.
- Live analytics: class distribution, prediction timeline, feature averages, probability summary, and a recent predictions table.

## Notes

- Both apps expect input features that match the "Multiclass Diabetes Dataset" schema and the engineered features defined in the notebook.
- The shared utilities guarantee that preprocessing remains consistent between notebooks, Gradio, and Streamlit experiences.
- If you retrain a new model, keep the artifact structure identical (model + preprocessor + metadata) for seamless deployment.
