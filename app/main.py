from fastapi import FastAPI
from app.api.v1 import discussion  # discussion.py 안에 있는 router 불러오기
from dotenv import load_dotenv
import os
from app.eureka_client.client import register_with_eureka  # ✅ Eureka 등록 함수

# 환경변수 로딩
load_dotenv()

# 디버깅용 출력 (나중에 제거 가능)
print("✅ GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

app = FastAPI()

# discussion 라우터 등록
app.include_router(discussion.router, prefix="/api/v1", tags=["discussion"])

@app.on_event("startup")
async def startup_event():
    # 동기 함수는 await 없이 호출
    register_with_eureka()

@app.get("/")
def hello():
    return {"msg": "FastAPI is registered with Eureka!"}
