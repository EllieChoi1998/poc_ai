from ocrEngine import PaddleEngine
from llmEngine import Gemma3Engine
from imageConverter import PDFtoPNG
import os

# PaddleEngine 인스턴스 생성
ocrengine = PaddleEngine(use_gpu=True, lang="korean")
llmengine = Gemma3Engine()
imageconverter = PDFtoPNG('./data/original', './data/converted')


pdf_file_name="pdf_4_1.pdf"
source_type = "운용지시서" # or, "계약서"
# PDF to PNG Convert
converted_files = imageconverter.convert_one_pdf(pdf_file_name)

if source_type == "운용지시서":
    ocr_results = {}
    for file_path in converted_files:
        ocrresult = ocrengine.process_image(file_path)
        ocr_results[file_path] = ocrresult

    result_dic = {}
    for file_path, ocrresult in ocr_results.items():  # .items() 메서드 사용하여 키-값 쌍 반복
        llm_result = llmengine.run(ocrresult=ocrresult["text"], source=source_type)
        result_dic[file_path] = llm_result
    
    # result_dic 을 한개의 텍스트 파일에 전부 저장 
    output_dir = './data/results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(pdf_file_name))[0]}_운용지시서_결과.md')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path, result in result_dic.items():
            page_name = os.path.basename(file_path)
            f.write(f"## {page_name}\n\n")
            f.write(result)
            f.write("\n\n---\n\n")  # 페이지 구분선
    
    print(f"운용지시서 결과가 저장되었습니다: {output_file}")

elif source_type == "계약서":
    ocr_results = ""
    for file_path in converted_files:
        result = ocrengine.process_image(file_path)
        # 텍스트 추출 결과가 성공적인 경우에만 추가
        if result["success"]:
            ocr_results += result["text"] + "\n\n"  # 페이지 간 구분을 위한 줄바꿈 추가
    
    llm_result = llmengine.run(ocrresult=ocr_results, source=source_type)

    # llm_result를 텍스트 파일안에 담아서 저장
    output_dir = './data/results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(pdf_file_name))[0]}_계약서_결과.md')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(llm_result)
    
    print(f"계약서 결과가 저장되었습니다: {output_file}")