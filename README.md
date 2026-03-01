# USALivingGuide

A Flask-based single-page AI guide application focused on US living topics for newcomers.

## Features
- Guidance on visa, tax, SSN, banking, health, housing, rideshare, and more.
- Step-by-step AI-generated answers via Google Vertex AI (Gemini).
- Background blog content fetching to enrich prompt context.

## Requirements
- Python 3.10+
- `GOOGLE_CLOUD_PROJECT` environment variable (or `GCP_PROJECT` as a fallback)
- (Optional) `VERTEX_LOCATION` (default: `us-central1`)
- (Optional) `GEMINI_MODEL` (default: `gemini-1.5-flash`)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running (development)
```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export VERTEX_LOCATION="us-east4"
export GEMINI_MODEL="gemini-1.5-flash"
python app.py
```

The app runs at `http://localhost:5000` by default.

## Running (production)
```bash
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export VERTEX_LOCATION="us-east4"
export GEMINI_MODEL="gemini-1.5-flash"
export PORT=5000
gunicorn -b 0.0.0.0:${PORT} --workers 2 --threads 4 --preload app:app
```

## Docker (Cloud Run compatible)
```bash
docker build -t usa-living-guide .
docker run --rm -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT="your-gcp-project-id" \
  -e VERTEX_LOCATION="us-east4" \
  -e GEMINI_MODEL="gemini-1.5-flash" \
  usa-living-guide
```

The Dockerfile is included in the root directory and launches via `gunicorn` using the `PORT` env variable for Cloud Run compatibility.

## Logging
By default, the application logs to stdout only (suitable for Docker/Cloud Run).
To enable file logging, set the `LOG_DIR` environment variable:
```bash
export LOG_DIR=logs
```

## Notes
- Set only one project env var: `GOOGLE_CLOUD_PROJECT` is preferred; `GCP_PROJECT` is the fallback.
- If Vertex AI is not configured, the API response runs in fallback summary mode.
- If the external blog source cannot be fetched, the app continues with fallback text.
- Feedback data is stored in memory only and cleared on restart (limited to the last 500 entries).

## Cloud Run Environment Variables
If `GOOGLE_CLOUD_PROJECT` or permissions are missing, the app returns a fallback summary instead of AI responses. For full AI answers, configure the service account permissions and env values:

```bash
gcloud run services update usa-living-guide \
  --region us-east4 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=your-gcp-project-id,VERTEX_LOCATION=us-east4,GEMINI_MODEL=gemini-1.5-flash
```

Also grant at least `roles/aiplatform.user` to the Cloud Run service account.

## Additional Features
- Use the **Ask Follow-up** field on answer cards to dig deeper into the current response.
- Submit messages + optional contact info from the **Feedback** tab (`/feedback`).
- Feedback is stored in memory and cleared on restart (limited to the last 500 entries).
