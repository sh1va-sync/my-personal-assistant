# Deploying Sync API

## Recommended hosting

Deploy the FastAPI service as a Docker web service. The included
`render.yaml` is configured for Render; the same `Dockerfile` works on Railway
and Cloud Run.

The public portfolio frontend should call the deployed API URL. Streamlit is
for local/internal use and is not part of the production API container.

## Required production environment variables

Set these in the hosting provider's secret/environment settings:

```env
ENVIRONMENT=production

GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

CHROMA_MODE=cloud
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
CHROMA_HOST=api.trychroma.com
CHROMA_COLLECTION=shiva_knowledge_production

FRONTEND_URL=https://your-frontend.example
ALLOWED_ORIGINS=https://your-frontend.example

REQUEST_TIMEOUT_SECONDS=120
PROVIDER_RETRY_ATTEMPTS=2
RATE_LIMIT_PER_MINUTE=40
DEFAULT_TIMEZONE=Asia/Kolkata
```

Do not deploy `.env`, API keys, `data/vectorstore`, or private knowledge files.

## First deployment

1. Create the production Chroma Cloud collection/database.
2. Configure the production variables above.
3. Build and deploy the Docker service.
4. In an environment with the production Chroma variables, run:

```powershell
uv run python scripts/ingest_knowledge.py
```

5. Verify:

```text
GET https://your-api.example/api/health
GET https://your-api.example/api/knowledge
```

The health response must report `status: "ok"` and `knowledge_ready: true`.

## Updating RAG knowledge

Edit a file under `data/knowledge`, then run the ingestion script with the
production Chroma variables. Ingestion replaces the configured collection, so
always verify `CHROMA_COLLECTION` before running it. The running API does not
need to be redeployed for a knowledge-only change.

For automation, run the ingestion script from a protected CI job using a
separate Chroma ingestion key. Never expose that key to the browser.

## Deployment checks

The service should:

- listen on `0.0.0.0` and the provider's `PORT`
- pass `/api/health`
- report a non-zero knowledge count
- answer `/api/chat`
- preserve `ALLOWED_ORIGINS` as the exact frontend origin
- keep Gemini and Chroma credentials only in provider secrets
