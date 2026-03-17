"""Vercel serverless entrypoint.

Vercel routes all requests to this module (see `vercel.json`).
Expose the FastAPI app object so `@vercel/python` can serve it.
"""

from app.main import create_app

app = create_app()

