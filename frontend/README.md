# Resume Match Frontend

A minimal Vite + React UI that talks to the existing FastAPI application exposed by `app/main.py`. The UI lets users paste a resume, forwards it to `/resume-match`, and renders the ranked job matches returned by the backend.

## Getting started

1. **Install dependencies**

   ```bash
   cd frontend
   npm install
   ```

2. **Expose the FastAPI server**

   Launch the backend from the project root (example command):

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

3. **Configure the API base URL (optional)**

   The UI defaults to `http://127.0.0.1:8000`. Override it with a `.env` file in `frontend/` if your FastAPI server lives elsewhere.

   ```bash
   echo "VITE_API_BASE_URL=http://localhost:8000" > .env
   ```

4. **Run the dev server**

   ```bash
   npm run dev
   ```

   Visit the address shown in the terminal (default: `http://localhost:5173`).

## Building for production

```bash
npm run build
npm run preview # optional local preview of the static bundle
```

The build step outputs static assets under `frontend/dist` that can be hosted by any static file server or reverse-proxied through FastAPI/NGINX.
