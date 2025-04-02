# document_processor.py
from ocrEngine import PaddleEngine
from llmEngine import Gemma3Engine
from imageConverter import PDFtoPNG
import os
from typing import List, Dict, Any, Optional


class DocumentProcessor:
    def __init__(self, original_dir: str = './data/original', 
                 converted_dir: str = './data/converted',
                 results_dir: str = './data/results',
                 use_gpu: bool = True,
                 language: str = "korean"):
        """
        Initialize the document processor with necessary components
        
        Args:
            original_dir: Directory containing original PDF files
            converted_dir: Directory for converted PNG files
            results_dir: Directory for processing results
            use_gpu: Whether to use GPU for OCR
            language: Language for OCR processing
        """
        self.original_dir = original_dir
        self.converted_dir = converted_dir
        self.results_dir = results_dir
        
        # Initialize components
        self.ocr_engine = PaddleEngine(use_gpu=use_gpu, lang=language)
        self.llm_engine = Gemma3Engine()
        self.image_converter = PDFtoPNG(original_dir, converted_dir)
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def process_document(self, pdf_filename: str, source_type: str) -> str:
        """
        Process a PDF document based on its type
        
        Args:
            pdf_filename: Name of the PDF file to process
            source_type: Type of document ("운용지시서" or "계약서")
            
        Returns:
            Path to the output result file
        """
        # Convert PDF to PNG
        converted_files = self.image_converter.convert_one_pdf(pdf_filename)
        
        # Process based on document type
        if source_type == "운용지시서":
            return self._process_operation_instruction(pdf_filename, converted_files)
        elif source_type == "계약서":
            return self._process_contract(pdf_filename, converted_files)
        else:
            raise ValueError(f"Unsupported document type: {source_type}")
    
    def _process_operation_instruction(self, pdf_filename: str, converted_files: List[str]) -> str:
        """Process operation instruction document type"""
        # Extract text from each page using OCR
        ocr_results = {}
        for file_path in converted_files:
            ocr_result = self.ocr_engine.process_image(file_path)
            ocr_results[file_path] = ocr_result
        
        # Process each page with LLM
        result_dic = {}
        for file_path, ocr_result in ocr_results.items():
            llm_result = self.llm_engine.run(ocrresult=ocr_result["text"], source="운용지시서")
            result_dic[file_path] = llm_result
        
        # Save results to file
        base_filename = os.path.splitext(os.path.basename(pdf_filename))[0]
        output_file = os.path.join(self.results_dir, f'{base_filename}_운용지시서_결과.md')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for file_path, result in result_dic.items():
                page_name = os.path.basename(file_path)
                f.write(f"## {page_name}\n\n")
                f.write(result)
                f.write("\n\n---\n\n")  # Page separator
        
        return output_file
    
    def _process_contract(self, pdf_filename: str, converted_files: List[str]) -> str:
        """Process contract document type"""
        # Combine all text from all pages
        ocr_results = ""
        for file_path in converted_files:
            result = self.ocr_engine.process_image(file_path)
            # Only add if successful
            if result["success"]:
                ocr_results += result["text"] + "\n\n"
        
        # Process combined text with LLM
        llm_result = self.llm_engine.run(ocrresult=ocr_results, source="계약서")
        
        # Save result to file
        base_filename = os.path.splitext(os.path.basename(pdf_filename))[0]
        output_file = os.path.join(self.results_dir, f'{base_filename}_계약서_결과.md')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(llm_result)
        
        return output_file