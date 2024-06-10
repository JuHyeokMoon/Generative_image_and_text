from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    set_seed
)
from diffusers import AutoPipelineForText2Image
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

import torch
import time
import copy
import gc
from pydantic import BaseModel
from enum import Enum
import random
import requests

from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


def trans_module(trans_model_select, text_list, lang="en"):
    trans_results = []
    if trans_model_select == "gugugo":
        trans_model = LLM(model="squarelike/Gugugo-koen-7B-V1.1-AWQ", download_dir="/data01/workspace/NPD/huggingface_models", quantization="awq", dtype="half")
        for text in text_list :
            if lang == "ko":
                prompt = f"### 한국어: {text}</끝>\n### 영어:"
            if lang == "en":
                prompt = f"### 영어: {text}</끝>\n### 한국어:"
                
            sampling_params = SamplingParams(temperature=0.01, stop=["</끝>"], max_tokens=10000)
                
            outputs = trans_model.generate(prompt, sampling_params)   
            
            trans_txt = ""
            for output in outputs:
                trans_txt += (output.outputs[0].text)
    
            trans_results.append(trans_txt)           

        destroy_model_parallel()
        del trans_model     
        gc.collect()        
        torch.cuda.empty_cache()
    
    if trans_model_select == "papago":
        # 사용량 확인   https://developers.naver.com/apps/#/myapps/BoiLvuJI9iO_28coG79t/overview
        client_id = "BoiLvuJI9iO_28coG79t"
        client_secret = ""
        url = "https://openapi.naver.com/v1/papago/n2mt"
        headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
        if lang == "ko":
            data = {
                "source": "ko",  # 원본 언어 (한국어)
                "target": "en",  # 번역할 언어 (영어)
                "text": text,
            }
        if lang == "en":
            data = {
                "source": "en",  # 번역할 언어 (영어)
                "target": "ko",  # 원본 언어 (한국어)
                "text": text,
            }
        response = requests.post(url, headers=headers, data=data)
        result = response.json()
        trans_txt = result["message"]["result"]["translatedText"]
  
    return trans_results


def save_image(image, random_int):
    file_name = f"{random_int}.png"
    image.save(f'/data01/workspace/NPD/img_results/{file_name}')
    return file_name


def generate_image(text:str, num_images:int):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", cache_dir="/data01/workspace/NPD/huggingface_models", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    total_file_names = []
    
    random_int = int(time.time())
    for i in range(num_images):
        generator = torch.manual_seed(random_int+i)

        image = pipe(
            prompt=text, 
            num_inference_steps=1, 
            guidance_scale=0.0,
            generator=generator
            ).images[0]

        file_name = save_image(image, random_int=random_int+i)       # 이미지 저장
        total_file_names.append(file_name)

    destroy_model_parallel()
    del pipe                    # 모델에 대한 참조를 삭제
    gc.collect()                # 가비지 컬렉션 수행, 메모리 반환
    torch.cuda.empty_cache()    # GPU 메모리 반환
    return total_file_names


def generate_text(prompt_list: str):
    # # 텍스트 생성 모델 초기화 (LLM MODEL)   # Deci/DeciLM-7B , mistralai/Mistral-7B-v0.1 , Xwin-LM/Xwin-LM-7B-V0.2
    txt_gen_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir="/data01/workspace/NPD/huggingface_models", device_map='auto')
    txt_gen_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", cache_dir="/data01/workspace/NPD/huggingface_models", device_map='auto')

    gen_results = []
    for text in prompt_list :
        
        encodeds = txt_gen_tokenizer.apply_chat_template(text, return_tensors="pt")
        model_inputs = encodeds.to('cuda')
        # model.to(device)

        generated_ids = txt_gen_model.generate(model_inputs, max_new_tokens=500, do_sample=True, pad_token_id=txt_gen_tokenizer.eos_token_id)
        decoded = txt_gen_tokenizer.batch_decode(generated_ids)
        
        # inputs = txt_gen_tokenizer.encode(text, return_tensors="pt").to("cuda")
        # outputs = txt_gen_model.generate(inputs, max_new_tokens=500, do_sample=True, top_p=0.95)
        # gen_copy_text = txt_gen_tokenizer.decode(outputs[0])
        
        # inputs = txt_gen_tokenizer(text, return_tensors="pt")
        # samples = txt_gen_model.generate(**inputs, max_new_tokens=300, temperature=0.2) # temperature 값이 낮을 수록 일관성있는 문장, 높을 수록 다양한 문장
        # gen_text = txt_gen_tokenizer.decode(samples[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        gen_copy_text = copy.deepcopy(decoded[0])
        
        s=gen_copy_text.rfind("[/INST]")
        e=gen_copy_text.rfind("</s>")

        gen_copy_text = gen_copy_text[s+7:e]
        
        gen_results.append(gen_copy_text)
        
    destroy_model_parallel()
    del txt_gen_model          
    del txt_gen_tokenizer      
    gc.collect()              
    torch.cuda.empty_cache()
    return gen_results

# def create_template(Main_Category, Sub_Category, Materials, Concepts):
#     return f"""
#     Category: {Main_Category} / {Sub_Category}
#     Materials: {Materials}
#     Concepts: {Concepts}"""


class Result(BaseModel):
    image_name: list
    text: list

class Category_1(str, Enum):
    Bakery="베이커리"
    icecream="아이스크림"
    beverages="음료"

class Category_2(str, Enum):
    Desserts="디저트"
    cakes="케이크"
    bread="식빵" 
    ice_cream_cones="아이스크림 콘"
    coffee="커피"
    fresh_fruit_juice="생과일 주스"
    shakes="쉐이크"


@app.post("/translation")
async def translation(Main_Category:Category_1, Sub_Category:Category_2, Materials: str, Concepts: str):
    total_start_time = time.time()
    num_images = 3
    trans_model_select = "gugugo"  # papago, gugugo    
    trans_model_select = trans_model_select.lower()

    Main_Category = Main_Category.name
    Sub_Category = Sub_Category.name
    
    # 번역 한글 -> 영어
    text_list = [Materials, Concepts]
    
    trans_text_list = trans_module(trans_model_select, text_list, lang="ko")
    
    Materials = trans_text_list[0]
    Concepts = trans_text_list[1]

    print(f"Materials 번역 결과 : {Materials}")
    print(f"Concepts 번역 결과 : {Concepts}")
    print('##########################################################################################################################################################')
    start_time = time.time()
    
    # 이미지 생성
    img_Materials = Materials.replace(",", " ") #! 
    img_prompt = f"Close up, {Main_Category}, Cinematic Light, Studio, {img_Materials}, advertisement photo style by Andrea van der Spuy" 
    image_name = generate_image(img_prompt, num_images)
    
    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f"이미지 생성 시간: {int(minutes)}분 {int(seconds)}초")
    print('##########################################################################################################################################################')
    start_time = time.time()

    warmup_template = """You're a product planner at a trendy food company. Given the following conditions, please answer using them as a guide. ['Categories': 'Bakery, Desserts', 'Ingredients': 'Cherry, Chocolate', 'Concept': 'Natural flavors, luxurious experiences, small luxuries']."""
    template = f"""You're a product planner at a trendy food company. Given the following conditions, please answer using them as a guide. ['Categories': '{Main_Category}, {Sub_Category}', 'Ingredients': '{Materials}', 'Concept': '{Concepts}']."""
    
    Product_Planning_Background = [
    {"role": "user", "content": f"{warmup_template} Write about product planning background the unique characteristics of your product, and its position in the marketplace with quantitative evidence. Write a 4-5 sentence post about product planning background."},
    {"role": "assistant", "content": "PRODUCT PLANNING BACKGROUND : Our new product falls under the bakery and dessert category, using natural cherries and chocolate as the primary ingredients. This product aims to prioritize the taste of nature and provide consumers with a luxurious experience and small luxuries. To understand our product's position in the market, we analyzed using various quantitative evidences. Through this, we gained confidence that this product will occupy a unique position in the premium dessert market. Based on this background, we developed a luxurious cherry chocolate dessert, and this product boasts characteristics that are differentiated from any other product."},
    {"role": "user", "content": f"{template} Write about product planning background the unique characteristics of your product, and its position in the marketplace with quantitative evidence. you write 3-4 sentence article about product planning background."}
    ]

    Why_we_planning_it = [
    {"role": "user", "content": f"{warmup_template} Write about the need for this product in the market and how you came up with the idea based on quantitative evidence. Write a 4-5 sentence post about reason for planning."},
    {"role": "assistant", "content": "REASON FOR PLANNING : Let me share the reasons why this product is needed in the market and how we came up with the idea based on quantitative evidence. Through recent data analysis, we noticed a trend where consumers prefer natural flavors and are seeking luxurious experiences and small luxuries. Therefore, we decided to develop a product in the bakery and dessert category using popular ingredients like cherry and chocolate, offering natural flavors and a luxurious experience. This plan is primarily focused on meeting the needs of consumers."},
    {"role": "user", "content": f"{template} Write a reason for planning based on quantitative evidence about why you're planning this product and why the market needs it. Write a 3-4 sentence post about reason for planning."}
    ]
    
    Key_target_audiences = [
    {"role": "user", "content": f"{warmup_template} Write a key audience analysis that analyzes your product's primary target audience and the need they have for your product, along with quantitative evidence of how your product can meet their needs. Write a 4-5 sentence post about target audience."},
    {"role": "assistant", "content": "TARGET AUDIENCE : Our primary target audience for this product is discerning consumers who appreciate the blend of natural flavors and luxurious experiences, as indicated by recent market research trends. They have shown a strong preference for bakery and dessert items that incorporate cherry and chocolate, seeking a small but luxurious indulgence. Quantitative evidence, such as market data and purchase behavior analysis, indicates that our product, which combines these preferred ingredients and concepts, can effectively meet these needs. Therefore, our product is well-positioned to satisfy this audience's desire for natural, luxurious treats."},
    {"role": "user", "content": f"{template} Analyze what your key customers want from your product and write about how you can satisfy their needs. Write a 3-4 sentence post about target audience."}
    ]
    
    Product_Features = [
    {"role": "user", "content": f"{warmup_template} Write product features with quantitative evidence of your product's key features, benefits, and how it differs from the competition. Write a 4-5 sentence post about product features."},
    {"role": "assistant", "content": "PRODUCT FEATURES : Our product, a unique blend of cherry and chocolate, stands out in the bakery and dessert category for its emphasis on natural flavors and providing a luxurious experience. It's not just a dessert, but a small luxury, crafted to cater to the discerning tastes of consumers. Based on market analysis, 70% of consumers prefer natural flavors, and our product meets this demand by using only natural cherry and chocolate. Moreover, unlike some competitors who use artificial flavorings, our commitment to natural ingredients sets us apart, offering a unique and superior taste experience."},
    {"role": "user", "content": f"{template} Write product features with quantitative evidence of your product's key features, benefits, and how it differs from the competition. Write a 3-4 sentence post about product features."}
    ]
    
    Product_assemblies_and_promotional_points = [
    {"role": "user", "content": f"{warmup_template} Focus on the product concept and promotion, and create product concepts and promotion points that reflect customer choices and market trends. Write a 4-5 sentence post about product concept and marketing highlights."},
    {"role": "assistant", "content": "PRODUCT CONCEPT AND MARKETING HIGHLIGHTS : Our product is designed with the concept of 'natural flavors meet small luxuries,' providing a luxurious experience through the blend of cherry and chocolate in bakery and dessert items. This concept is in line with current market trends, as consumers increasingly seek natural ingredients and indulgent experiences. Our promotional points will highlight the luxurious experience of our product, and its natural, high-quality ingredients. We aim to attract consumers by emphasizing the unique, premium experience our product offers, setting a new standard in the bakery and dessert category."},
    {"role": "user", "content": f"{template}Focus on the product concept and promotion, and create product concepts and promotion points that reflect customer choices and market trends. Write a 3-4 sentence post about product concept and marketing highlights."}
    ]
    
    prompt_list = [Product_Planning_Background, Why_we_planning_it, Key_target_audiences, Product_Features, Product_assemblies_and_promotional_points]
    # 텍스트 생성
    set_seed(int(time.time()))
    gen_text = generate_text(prompt_list)

    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f"생성 결과 : {gen_text}")
    print(f"텍스트 생성 시간 : {int(minutes)}분 {int(seconds)}초")
    print('##########################################################################################################################################################')
    start_time = time.time()

    # 번역 영어 -> 한글
    for i in range(len(gen_text)) :      # 개행 문자 치환 후 다시 '\n'변경
        gen_text[i] = gen_text[i].replace("\n","@&")  # 개행 문자 치환 후 다시 '\n'변경
    chat_result = trans_module(trans_model_select, gen_text, lang="en")
    
    for i in range(len(chat_result)) :      # 개행 문자 치환 후 다시 '\n'변경
        chat_result[i] = chat_result[i].replace("@&","\n")  

    end_time = time.time()
    minutes, seconds = divmod(end_time - start_time, 60)
    print(f"번역 결과 : {chat_result}")
    print(f"번역 소요 시간 : {int(minutes)}분 {int(seconds)}초")
    print('##########################################################################################################################################################')

    # 이미지 파일 이름, 텍스트 생성(번역) 결과 담기
    # result = Result(image_name=["1.jpg","2.jpg"], text=chat_result)
    result = Result(image_name=image_name, text=chat_result)

    total_end_time = time.time()
    minutes, seconds = divmod(total_end_time - total_start_time, 60)
    print(f"총 소요 시간 : {int(minutes)}분 {int(seconds)}초")
    print('##########################################################################################################################################################')

    return result

@app.get("/images")
async def get_image(file_name: str):
    return FileResponse(f'/data01/workspace/NPD/img_results/{file_name}')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 0,0,0,0 localhost
