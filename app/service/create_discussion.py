# 파일 저장 및 불러오기용 (토큰 저장)
import json
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()  # ✅ 환경변수 자동 로딩
import os
from app.core.llm_client import get_llm_client,get_langchain_llm
from app.core.discussion_prompt import Prompt
from pydantic import BaseModel


################################################################
def summation(input_data):

    description = input_data
    print("🧪 Sending to LLM:\n", description)  # 👉 확인 필수       
    
    class SummationResponse(BaseModel):
        input: str
        output: str


    llm = get_langchain_llm(is_lightweight=False)

    parser = JsonOutputParser(pydantic_object=SummationResponse)


    system_prompt=Prompt.discussion_prompt()

    prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt ),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        print("summation_response\n",json.dumps(result, indent=2,ensure_ascii=False))
        
        return result
    
        
    response = parse_product(description)
    print("summation_response\n",response)
   


    return response
################################################################


