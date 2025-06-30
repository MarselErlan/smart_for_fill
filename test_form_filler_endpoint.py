import requests
import time
import subprocess
import os
import pytest
from loguru import logger

BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{BASE_URL}/api/fill-form-in-browser"
TEST_URL = "https://jobs.ashbyhq.com/wander/96d09210-9d42-4e67-a035-00573f4426d9/application?departmentId=dd723511-a55f-43a7-8932-e3d77be5e4a4"

def start_server_process():
    """Starts the Uvicorn server as a separate process."""
    server_command = [
        "venv310/bin/uvicorn",
        "main:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload"
    ]
    server_process = subprocess.Popen(server_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info("Starting FastAPI server with reload...")

    # Wait for the server to start
    retries = 20  # Increased retries for slower startups
    is_ready = False
    for i in range(retries):
        try:
            time.sleep(3) # Give it a moment before polling
            response = requests.get(f"{BASE_URL}/status", timeout=5)
            if response.status_code == 200:
                logger.info("Server started successfully.")
                is_ready = True
                break
        except requests.ConnectionError:
            logger.warning(f"Attempt {i+1}: Connection failed. Server not ready yet.")
        except requests.ReadTimeout:
            logger.warning(f"Attempt {i+1}: Read timeout. Server might be slow to start.")

    if not is_ready:
        stdout, stderr = server_process.communicate()
        logger.error("Server failed to start.")
        logger.error(f"STDOUT: {stdout.decode()}")
        logger.error(f"STDERR: {stderr.decode()}")
        server_process.terminate()
        pytest.fail("Server did not start within the allotted time.")

    return server_process

def test_fill_form_in_browser_success():
    """
    Test the /api/fill-form-in-browser endpoint for a successful scenario.
    """
    server_process = None
    try:
        server_process = start_server_process()
        
        logger.info(f"Testing endpoint: {API_URL}")
        payload = {
            "url": TEST_URL,
            "user_data": {},
            "headless": True
        }
        
        # Increased timeout for this long-running task
        response = requests.post(API_URL, json=payload, timeout=300)
        
        logger.info(f"Response status code: {response.status_code}")
        try:
            response_json = response.json()
            logger.info(f"Response JSON: {response_json}")
            assert "status" in response_json
            assert response_json["status"] == "success"
            assert "result" in response_json
            assert "steps" in response_json["result"]
            assert "form_filling" in response_json["result"]["steps"]
            assert response_json["result"]["steps"]["form_filling"]["status"] == "success"
            
        except (requests.exceptions.JSONDecodeError, KeyError, AssertionError) as e:
            logger.error(f"Failed to parse or validate response: {e}")
            logger.error(f"Raw response text: {response.text}")
            pytest.fail(f"Response validation failed: {e}")

        assert response.status_code == 200


    finally:
        if server_process:
            logger.info("Stopping FastAPI server...")
            server_process.terminate()
            server_process.wait()
            logger.info("Server stopped.")

if __name__ == "__main__":
    pytest.main([__file__]) 