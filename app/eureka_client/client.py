import socket
import requests
import uuid
import os
from app.eureka_client.heartbeat import start_heartbeat
from app.config.eureka_config import APP_NAME, PORT, INSTANCE_ID, EUREKA_URL

EUREKA_URL = os.getenv("EUREKA_URL", "http://localhost:8761/eureka")
APP_NAME = os.getenv("APP_NAME", "EUM-DISCUSSION-ROOM")
PORT = int(os.getenv("PORT", 8000))
INSTANCE_ID = f"{APP_NAME}:{uuid.uuid4()}"

def register_with_eureka():
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    data = {
        "instance": {
            "instanceId": INSTANCE_ID,
            "hostName": hostname,
            "app": APP_NAME.upper(),
            "ipAddr": ip,
            "vipAddress": APP_NAME,
            "status": "UP",
            "port": {"$": PORT, "@enabled": True},
            "dataCenterInfo": {
                "@class": "com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo",
                "name": "MyOwn"
            }
        }
    }

    url = f"{EUREKA_URL}/apps/{APP_NAME.upper()}"
    try:
        res = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        print(f"[Eureka] 등록 결과: {res.status_code}")
    except Exception as e:
        print(f"[Eureka] 등록 실패: {e}")

    start_heartbeat()