import requests
import threading
import time
from app.config.eureka_config import APP_NAME, INSTANCE_ID, EUREKA_URL

def start_heartbeat():
    def heartbeat():
        while True:
            try:
                url = f"{EUREKA_URL}/apps/{APP_NAME.upper()}/{INSTANCE_ID}"
                requests.put(url)
                print("[Eureka] Heartbeat sent")
            except Exception as e:
                print(f"[Eureka] Heartbeat failed: {e}")
            time.sleep(30)

    threading.Thread(target=heartbeat, daemon=True).start()
