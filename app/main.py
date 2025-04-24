from fastapi import FastAPI
from app.api.v1 import discussion  # discussion.py 안에 있는 router 불러오기
from dotenv import load_dotenv
import os
load_dotenv()  # ✅ 환경변수 자동 로딩

# 디버깅용 출력 (나중에 삭제해도 됨)
print("✅ GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

app = FastAPI()

# 라우터 등록
app.include_router(discussion.router, prefix="/api/v1", tags=["discussion"])


