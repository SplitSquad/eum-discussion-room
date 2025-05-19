from fastapi import FastAPI
from app.api.v1 import discussion
from dotenv import load_dotenv
import os

from py_eureka_client import eureka_client
from app.config.eureka_config import APP_NAME, PORT, EUREKA_URL

import asyncio

# 환경변수 로딩
load_dotenv()

print("✅ GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

app = FastAPI()

# 라우터 등록
app.include_router(discussion.router, prefix="/api/v1", tags=["discussion"])

@app.on_event("startup")
async def startup_event():
    # ✅ 비동기 Eureka 등록
    await eureka_client.init_async(
        eureka_server=EUREKA_URL,
        app_name=APP_NAME,
        instance_port=PORT,
        instance_ip="127.0.0.1",
        health_check_url=f"http://localhost:{PORT}/",
        home_page_url=f"http://localhost:{PORT}/",
        status_page_url=f"http://localhost:{PORT}/status",
        renewal_interval_in_secs=30,
        duration_in_secs=90,
    )

@app.get("/")
def hello():
    return {"msg": "FastAPI is registered with Eureka!"}
