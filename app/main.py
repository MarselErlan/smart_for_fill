"""
Script to run the Smart Form Fill API server using uvicorn
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default port
PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    print(f"Starting Smart Form Fill API on port {PORT}")
    print(f"API documentation available at http://localhost:{PORT}/api/docs")
    uvicorn.run("app.api:app", host="0.0.0.0", port=PORT, reload=True) 