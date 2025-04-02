# process_local.py
"""
Command-line script for processing documents locally without running the API server
"""
import argparse
from document_processor import DocumentProcessor

def main():
    parser = argparse.ArgumentParser(description="Process a PDF document with OCR and LLM")
    parser.add_argument("--pdf", required=True, help="PDF filename to process")
    parser.add_argument("--type", required=True, choices=["운용지시서", "계약서"], 
                        help="Document type (운용지시서 or 계약서)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DocumentProcessor(use_gpu=args.gpu)
    
    # Process document
    try:
        result_file = processor.process_document(args.pdf, args.type)
        print(f"처리가 완료되었습니다. 결과 파일: {result_file}")
    except Exception as e:
        print(f"처리 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()