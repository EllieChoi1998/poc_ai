# Gemma3 Engine

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch

class Gemma3Engine():
    def __init__(self):
        model_id = "google/gemma-3-4b-it" 
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def run_model(self, ocrtext, prompt):
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ocrtext},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded
    
    def run(self, ocrresult, source):
        if source == "운용지시서":
            # 일단 테이블 위치 판독하는 부분은 미개발 상태라서 프롬포트를 다르게 설정함. (단순 파이프라인 테스트용용)
            # prompt = "표 속 텍스트를 ocr한 결과야. 해당 내용을 마크다운으로 사람이 읽을 수 있게 정리해줘." 
            prompt = "운용지시서를 ocr한 결과야. 해당 내용을 마크다운으로 사람이 읽을 수 있게 정리해줘."
            return self.run_model(ocrtext=ocrresult, prompt=prompt)
        elif source == "계약서":
            prompt = "위 계약서 내용을 마크다운으로 사람이 읽을 수 있게 정리해줘."

            cropped_ocr_text = self.crop(ocrresult, length=3000)  # 적절한 길이 값 지정
            llm_result = ""
            for ocrtext in cropped_ocr_text:
                llm_result += (self.run_model(ocrtext=ocrtext, prompt=prompt))
            return llm_result

    def crop(self, ocrresult, length):
        if len(ocrresult) <= length:
            return [ocrresult]  # 짧은 텍스트는 그대로 반환
            
        cropped_ocr_text = []
        i = 0
        while i < len(ocrresult):
            # 첫 번째 청크는 처음부터 length까지
            if i == 0:
                cropped_ocr_text.append(ocrresult[i:i+length])
            else:
                # 이전 청크와 일부 중복되도록 설정 (문맥 유지를 위해)
                overlap = min(500, length//4)  # 중복 정도를 조절할 수 있는 값
                start = i - overlap
                end = min(i + length - overlap, len(ocrresult))
                cropped_ocr_text.append(ocrresult[start:end])
            
            # 다음 시작점은 중복 없이 진행
            i += (length - overlap if i > 0 else length)
            
            # 마지막 조각인 경우 반복 종료
            if i >= len(ocrresult):
                break
        
        return cropped_ocr_text