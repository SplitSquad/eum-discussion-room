# 파일 저장 및 불러오기용 (토큰 저장)
import json
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()  # ✅ 환경변수 자동 로딩
import os

# 디버깅용 출력 (나중에 삭제해도 됨)
print("✅ GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

################################################################
def summation(input_data):
    
    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
    )
    parser = JsonOutputParser(pydantic_object={
        "type": "object",
        "properties": {
            "input": {"type": "string"},
            "output": {"type": "string"},
        }
    })
    prompt = ChatPromptTemplate.from_messages([
    ("system", """
        1. Its role is to summarize news articles.
        2. I will provide the title, date, and content.
        3. Summarize the content 
        default. Answer just one. Please respond in this json format. 
     
        "title": " ",
        "date": " ",
        "content": " "
  
    """),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        
        return json.dumps(result, indent=2,ensure_ascii=False)
        
    description = input_data
    response = parse_product(description)
   

    response = json.loads(response)  # 문자열 → 딕셔너리
    print("summation_response\n",response)

    return response
################################################################


################################################################
def discussion(summary) : 

    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
    )
    parser = JsonOutputParser(pydantic_object={
        "type": "object",
        "properties": {
            "input": {"type": "string"},
            "output": {"type": "string"},
        }
    })
    prompt = ChatPromptTemplate.from_messages([
    ("system", """
        1. The role is to change the article to fit the frame.
        2. I will provide the title, date, and content.
        3. If there is no article, a null value is provided.
        default. Respond **only** with the following JSON format In korean:

        "title": "<positive opinion + negative opinion>",
        "category": "<one of option:정치/사회, 경제, 생활/문화, 과학/기술, 스포츠, 엔터테인먼트>",
        "content": "content",
        "vote" : "
                What do you think about <content>?
                positive opinion - < > 
                negative opinion - < > "
    """),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        
        return json.dumps(result, indent=2,ensure_ascii=False)
        
    description = summary
    response = parse_product(description)
    
    response = json.loads(response)  # 문자열 → 딕셔너리
    print("discussion_response\n",response)

    return response
################################################################

input_data = """
    {
      "title": "[프로필] 마용주 신임 대법관…인청 때 '계엄 절차' 지적한 정통 법관",
      "date": "2025.04.08. 오전 11:28",
      "content": "현대차 사내하청업체 253명에게 '노동자 지위' 인정해 주목마용주 당시 대법관 후보자가 26일 서울 여의도 국회에서 열린 인사청문회에 출석해 의원 질의에 답하고 있다. 2024.12.26/뉴스1 ⓒNews1 안은나 기자(서울=뉴스1) 이세현 기자 = 한덕수 대통령 권한대행 국무총리가 8일 임명한 마용주 신임 대법관(56·사법연수원 23기)은 이론과 실무에 모두 능한 정통 법관으로 꼽힌다.특히 2017년부터 2021년까지 대법원 선임재판연구관과 수석재판연구관을 역임해 상고심 재판에 능통하고 법리 이해도가 높다는 평가를 받는다.경남 합천 태생으로 부산 낙동고를 졸업한 마 대법관은 서울대 법학과 4학년 재학 중 사법시험에 합격해 1997년 서울지법 판사로 임관했다.이후 27년간 서울·대전·통영·제주 등 전국 각지 법원에서 민사·형사·행정 등 다양한 재판을 경험했다.법원행정처 인사심의관·인사1심의관실 판사·윤리감사관 등 사법행정도 겸비해 엘리트 코스를 밟았다는 평가를 받는다.그는 법관의 친인척이 일하는 법무법인의 수임 사건 처리와 관련한 대법원 공직자윤리위원회의 권고 의견을 마련하고 법관의 외부 강의 기준을 확립하는 데 기여했다는 평도 듣는다.주요 사건 판결로는 서울중앙지법 부장판사 당시 현대자동차 사내하청업체 소속 253명에게 노동자 지위를 인정한 판결이 언급된다. 서울고법에서는 윤미향 전 의원의 '정의기억연대 후원금 횡령 사건'과 '백현동 로비스트 의혹' 당사자인 김인섭 씨의 항소심을 심리했다.마 대법관은 후보자 시절인 지난해 12월 26일 국회 인사청문회에서 윤석열 대통령의 12·3 비상계엄과 관련해 \"절차를 제대로 준수하지 못해 문제 제기할 수 있는 사안 같다\"고 밝혀 주목받았다.조희대 대법원장은 지난해 12월 27일 퇴임하는 김상환 대법관의 후임으로 마 대법관을 임명 제청했으며, 한 대행은 지난 4일 윤 대통령이 파면되자 본인이 직접 마 대법관을 임명했다.△1969년 △경남 합천 △서울대 법학과 졸업△ 제33회 사법시험 합격(연수원 23기) △서울지법 판사 △미국 조지타운대학 교육파견△법원행정처 인사관리심의관 △법원행정처 윤리감사관 △서울중앙지법 부장판사 △대법원 선임재판연구관 △대법원 수석재판연구관 △서울고법 부장판사"
    },
    {
      "title": "[그래픽] 신임 대법관 후보자 마용주 프로필",
      "date": "2024.11.26. 오후 6:01",
      "content": "(서울=연합뉴스) 김토일 기자 = 조희대 대법원장은 12월 27일 퇴임하는 김상환 대법관의 후임으로 마용주(55·사법연수원 23기) 서울고법 부장판사를 윤석열 대통령에게 임명제청했다.kmtoil@yna.co.kr페이스북tuney.kr/LeYN1 X(트위터) @yonhap_graphics"
    },
"""

# print("기사 내용요약!!")
# sum = summation(input_data)
# print(sum)

# print("\n\n\n\n토론방 생성")
# create = discussion(sum)
# print(create)
