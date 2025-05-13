import os
import uuid

APP_NAME = os.getenv("APP_NAME", "EUM-DISCUSSION-ROOM")
PORT = int(os.getenv("PORT", 8000))
INSTANCE_ID = f"{APP_NAME}:{uuid.uuid4()}"
EUREKA_URL = os.getenv("EUREKA_URL", "http://localhost:8761/eureka")
