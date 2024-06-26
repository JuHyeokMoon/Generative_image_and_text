# 생성형 AI 기반 상품 기획 플랫폼
상품 개발 시 AI 기술이 보조 역할을 할수 있도록, 상품 이미지와 상품 기획서를 제공하는 플랫폼 개발    

# 프로세스 요약
1. 사용자 입력(제품 키워드, 설명)
2. 한글 영문 번역
3. 이미지 생성을 위한 Prompt 정리
4. 이미지 생성 및 저장
5. 텍스트 생성을 위한 Prompt 정리
6. 텍스트 생성
7. 생성된 텍스트 번역
8. 생성된 이미지와 텍스트 전달

# 특이 사항
1. 제품 기획서와 같이 특정 양식의 텍스트 생성이 필요.  
   > 질문과 답에 대한 예시를 미리 보여 주어 출력을 원하는 방향으로 유도 할 수 있는 Instruct 모델 사용  

2. 모델 사이즈가 커서 서버를 띄울 때 GPU를 할당하여 모델을 Load 할 수 없음.  
   > 각 프로세스(번역, 이미지 생성, 텍스트 생성) 마다 모델을 Load 하고 추론 후 offload 하는 방식 선택.  
   > 추후 AWS 환경 세팅으로 모델들을 미리 Load 하여 불필요한 프로세스를 없애고 속도를 단축시킬 예정.
