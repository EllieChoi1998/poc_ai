# ai_server.py (포트 8001번 서버)
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
from document_processor import DocumentProcessor
import uvicorn
from typing import Optional
from pydantic import BaseModel

app = FastAPI(title="Document Processing API",
              description="API for processing PDF documents with OCR and LLM")

# 디렉토리 경로 설정
ORIGINAL_DIR = './data/original'
CONVERTED_DIR = './data/converted'
RESULTS_DIR = './data/results'

# 디렉토리 생성
os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(CONVERTED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 프로세서 초기화 (CPU 모드로 시작)
processor = DocumentProcessor(
    original_dir=ORIGINAL_DIR,
    converted_dir=CONVERTED_DIR,
    results_dir=RESULTS_DIR,
    # use_gpu=False  # GPU 메모리 문제로 기본값은 False
)

class ProcessResponse(BaseModel):
    success: bool
    message: str
    result_file: Optional[str] = None

@app.post("/process/", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    source_type: str = Form(...)  # "운용지시서" or "계약서"
):
    """
    Process a PDF document based on its type.
    
    - **file**: PDF file to process
    - **source_type**: Document type ("운용지시서" or "계약서")
    
    Returns the path to the result file
    """
    # Validate source type
    if source_type not in ["운용지시서", "계약서"]:
        raise HTTPException(status_code=400, detail="Invalid source type. Must be '운용지시서' or '계약서'")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file to original directory
    file_path = os.path.join(ORIGINAL_DIR, file.filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        result_file = processor.process_document(file.filename, source_type)
        
        # 전체 경로가 아닌 파일명만 반환
        result_filename = os.path.basename(result_file)
        
        return ProcessResponse(
            success=True,
            message=f"{source_type} 처리가 완료되었습니다.",
            result_file=result_filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/results/{filename}")
async def get_result(filename: str):
    """
    Download a specific result file.
    
    - **filename**: Name of the result file
    """
    file_path = os.path.join(RESULTS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(file_path, media_type="text/markdown", filename=filename)

@app.get("/")
def read_root():
    return {"message": "Document Processing API"}

if __name__ == "__main__":
    uvicorn.run("ai_server:app", host="0.0.0.0", port=8001, reload=True)