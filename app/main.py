from fastapi import FastAPI
from contextlib import asynccontextmanager
from py_eureka_client import eureka_client
from dotenv import load_dotenv
from os import getenv, path
from loguru import logger

# 환경 변수 로드
env_path = path.join(path.dirname(path.dirname(__file__)), '.env')
load_dotenv(env_path)

EUREKA_IP = getenv("EUREKA_IP", "http://localhost:8761/eureka")
EUREKA_APP_NAME = getenv("EUREKA_APP_NAME", "eum-classifier")
EUREKA_HOST = getenv("EUREKA_HOST", "localhost")
EUREKA_PORT = int(getenv("EUREKA_PORT", "8003"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[WORKFLOW] Server started successfully")
    logger.info("[EUREKA] Loading Eureka configuration...")
    logger.debug(f"[EUREKA] - Server: {EUREKA_IP}")
    logger.debug(f"[EUREKA] - App Name: {EUREKA_APP_NAME}")
    logger.debug(f"[EUREKA] - Host: {EUREKA_HOST}")
    logger.debug(f"[EUREKA] - Port: {EUREKA_PORT}")

    try:
        await eureka_client.init_async(
            eureka_server=EUREKA_IP,
            app_name=EUREKA_APP_NAME,
            instance_host=EUREKA_HOST,
            instance_port=EUREKA_PORT
        )
        logger.info("[EUREKA] ✅ Eureka client initialized successfully")
    except Exception as e:
        logger.error(f"[EUREKA] ❌ Eureka client initialization failed: {str(e)}")
        raise

    yield
    # 종료 시 필요한 정리 작업을 여기에 추가할 수 있습니다.
    logger.info("[EUREKA] 🔻 Application shutdown initiated")

app = FastAPI(lifespan=lifespan)


# # 1
# # 환경변수 로딩
# load_dotenv()

# print("✅ GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

# app = FastAPI()

# # 라우터 등록
# app.include_router(discussion.router, prefix="/api/v1", tags=["discussion"])

# @app.on_event("startup")
# async def startup_event():
#     # ✅ 비동기 Eureka 등록
#     await eureka_client.init_async(
#         eureka_server=EUREKA_URL,
#         app_name=APP_NAME,
#         instance_port=PORT,
#         instance_ip="0.0.0.0",
#         health_check_url=f"http://0.0.0.0:{PORT}/",
#         home_page_url=f"http://0.0.0.0:{PORT}/",
#         status_page_url=f"http://0.0.0.0:{PORT}/status",
#         renewal_interval_in_secs=30,
#         duration_in_secs=90,
#     )

# @app.get("/")
# def hello():
#     return {"msg": "FastAPI is registered with Eureka!"}
