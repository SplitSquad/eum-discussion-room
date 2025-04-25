from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Any, Dict
from app.service.create_discussion import summation, discussion
import json
from dotenv import load_dotenv
load_dotenv()  # ✅ 환경변수 자동 로딩


router = APIRouter()

# 요청 구조
class Article(BaseModel):
    title: str
    date: str
    content: str

# 응답 구조
class DiscussionResponse(BaseModel):
    discussion: Any  # discussion에서 dict 반환

@router.post("/discussion", response_model=DiscussionResponse)
async def generate_discussion(articles: List[Article]):
    try:
        # 1. 사용자 입력 데이터를 JSON 문자열로 직렬화
        input_json = json.dumps([article.dict() for article in articles], ensure_ascii=False)

        # 2. 기사 요약
        summary_result = summation(input_json)

        # 3. 토론방 생성
        discussion_result = discussion(summary_result)

        # 4. 응답
        return DiscussionResponse(
            discussion=discussion_result
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
