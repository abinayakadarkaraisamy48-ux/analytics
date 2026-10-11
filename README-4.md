# SmartSpend AI — Streamlit

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.

The app creates `smartspend.db` automatically. The ML model trains on generated synthetic data; SMS extraction is best-effort, so review the fields and category before saving.

## Deploy to Streamlit Community Cloud
Upload `app.py` and `requirements.txt` to a GitHub repository. In Streamlit Community Cloud, select the repository and set the main file path to `app.py`.

Note: local SQLite storage may not be durable across hosted restarts/redeployments. For multi-user production use, add authentication, user isolation and managed persistent storage.
