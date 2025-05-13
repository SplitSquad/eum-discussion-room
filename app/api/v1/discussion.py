from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional
from app.service.create_discussion import summation
import json
import traceback
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

MAX_SIZE = 1024 * 1024  # 1MB

# 요청 구조
class Article(BaseModel):
    title: str
    date: str
    content: str

# 응답 구조
class DiscussionResponse(BaseModel):
    discussion: Any

@router.post("/discussion", response_model=DiscussionResponse)
async def generate_discussion(articles: List[Optional[Article]]):
    try:
        valid_articles = [a.dict() for a in articles if a is not None]

        chunks = []
        current_chunk = []
        current_size = 0

        for article in valid_articles:
            article_json = json.dumps(article, ensure_ascii=False)
            size = len(article_json.encode('utf-8'))

            if current_size + size > MAX_SIZE:
                # 현재까지 모은 걸 하나의 chunk로 저장
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

            current_chunk.append(article)
            current_size += size

        # 마지막 조각 추가
        if current_chunk:
            chunks.append(current_chunk)

        # 모든 조각 병합해서 summation 수행
        merged_json = json.dumps([item for chunk in chunks for item in chunk], ensure_ascii=False)
        summary_result = summation(merged_json)

        return DiscussionResponse(discussion=summary_result)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
