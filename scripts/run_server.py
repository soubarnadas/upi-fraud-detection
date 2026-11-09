"""
Server launcher script for the UPI Fraud Detection scoring API.
"""

import uvicorn
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting UPI Fraud Detection API on port {port}...")
    uvicorn.run("src.service.api:app", host="0.0.0.0", port=port, reload=False)
