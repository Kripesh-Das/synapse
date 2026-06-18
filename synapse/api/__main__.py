"""Run the synapse API server.

Usage:
    python -m synapse.api
    # or
    uvicorn synapse.api.app:app --host 0.0.0.0 --port 8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "synapse.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
