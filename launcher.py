import subprocess
import requests
import time

# Launch backend fast api
backend_process = subprocess.Popen(["fastapi", "run", "fastapi/modelapi.py"])

# Wait until the backend is ready
while True:
    try:
        # Check if the backend is ready by sending a request to an endpoint
        response = requests.get("http://localhost:8000/")
        response.raise_for_status()
        if response.status_code == 200:
            break  # Backend is ready
    except requests.RequestException:
        pass  # Backend is not yet ready
    time.sleep(1)

# Launch frontend after backend is done
subprocess.run(["streamlit", "run", "streamlit/app.py"])
