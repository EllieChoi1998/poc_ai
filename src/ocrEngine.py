from paddleocr import PaddleOCR, draw_ocr
import os
from PIL import Image
import numpy as np
# TODO : Table Recognition 해서 테이블 좌표값들을 받아와서 해당 좌표값에 맞게 원본 이미지를 각 테이블별로 crop하는 로직 추가되면 좋을것 같음.
class PaddleEngine:
    def __init__(self, use_gpu=True, lang="korean", font_path='./paddleocr/korean.ttf'):
        """
        PaddleOCR 엔진 초기화
        
        Args:
            use_gpu (bool): GPU 사용 여부
            lang (str): 사용할 언어 (korean, en 등)
            font_path (str): 시각화에 사용할 폰트 경로
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.font_path = font_path
        self.ocr = PaddleOCR(use_gpu=self.use_gpu, lang=self.lang)
        self.output_dir = './workspace'
        
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 폰트 파일 존재 확인
        if not os.path.exists(self.font_path):
            print(f"경고: 폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다: {self.font_path}")
    
    def verify_image_path(self, img_path):
        """
        이미지 파일 경로 유효성 검증
        
        Args:
            img_path (str): 확인할 이미지 파일 경로
            
        Returns:
            bool: 파일 존재 여부
        """
        # 현재 작업 디렉토리 확인
        current_dir = os.getcwd()
        print(f"현재 작업 디렉토리: {current_dir}")
        
        # 이미지 파일이 존재하는지 확인
        if not os.path.exists(img_path):
            print(f"이미지 파일을 찾을 수 없습니다: {img_path}")
            
            # 디렉토리가 존재하는지 확인
            base_dir = os.path.dirname(img_path)
            if not os.path.exists(base_dir):
                print(f"디렉토리가 존재하지 않습니다: {base_dir}")
            
            # 가능한 이미지 파일 경로 제안
            possible_paths = [
                img_path,
                f'./{img_path}',
                f'../{img_path}',
                f'../../{img_path}',
                f'/home/dkzndk/work/ibk/{img_path}'
            ]
            
            print("다음 경로를 시도해보세요:")
            for path in possible_paths:
                if os.path.exists(path):
                    print(f"✅ 이 경로가 존재합니다: {path}")
                    return path
                else:
                    print(f"❌ 이 경로는 존재하지 않습니다: {path}")
            
            return False
        
        print(f"이미지 파일을 찾았습니다: {img_path}")
        return img_path
    
    def run_ocr(self, img_path):
        """
        이미지에서 OCR 실행
        
        Args:
            img_path (str): 이미지 파일 경로
            
        Returns:
            list: OCR 결과
            None: 실패 시
        """
        # 이미지 경로 확인
        valid_path = self.verify_image_path(img_path)
        if not valid_path:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        
        # OCR 실행
        result = self.ocr.ocr(valid_path)
        
        # 결과가 비어있는지 확인
        if not result or not result[0]:
            print("인식된 텍스트가 없습니다.")
            return None
        
        # 첫 번째 페이지 결과 가져오기 (PaddleOCR은 여러 페이지를 처리할 수 있음)
        result = result[0]
        
        return result, valid_path
    
    def print_ocr_results(self, result):
        """
        OCR 결과 출력
        
        Args:
            result (list): OCR 결과
        """
        print("인식된 텍스트:")
        for idx, line in enumerate(result):
            print(f"텍스트 {idx+1}: {line[1][0]} (신뢰도: {line[1][1]:.4f})")
    
    def visualize_result(self, img_path, result, output_name='test_result'):
        """
        OCR 결과 시각화 및 저장
        
        Args:
            img_path (str): 원본 이미지 경로
            result (list): OCR 결과
            output_name (str): 출력 파일 이름 (확장자 제외)
            
        Returns:
            str: 저장된 이미지 경로
        """
        # 시각화를 위한 준비
        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        
        # 결과 시각화 및 저장
        output_path = os.path.join(self.output_dir, f'{output_name}.jpg')
        
        im_show = draw_ocr(image, boxes, txts, scores, font_path=self.font_path)
        im_show = Image.fromarray(im_show)
        im_show.save(output_path)
        
        print(f"결과 이미지가 저장되었습니다: {output_path}")
        return output_path
    
    def save_text_result(self, result, output_name='ocr_result'):
        """
        인식된 텍스트를 파일로 저장
        
        Args:
            result (list): OCR 결과
            output_name (str): 출력 파일 이름 (확장자 제외)
            
        Returns:
            str: 저장된 텍스트 파일 경로
        """
        # 인식된 모든 텍스트를 텍스트 파일로 저장
        txt_path = os.path.join(self.output_dir, f'{output_name}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in result:
                f.write(f"{line[1][0]}\n")
        
        print(f"인식된 텍스트가 저장되었습니다: {txt_path}")
        return txt_path
    
    def get_text_from_result(self, result):
        """
        OCR 결과에서 텍스트만 추출하여 문자열로 반환
        
        Args:
            result (list): OCR 결과
            
        Returns:
            str: 추출된 텍스트
        """
        texts = [line[1][0] for line in result]
        return '\n'.join(texts)
    
    def process_image(self, img_path, output_base_name=None):
        """
        이미지 처리 전체 과정 실행 (OCR 실행, 시각화, 텍스트 저장)
        
        Args:
            img_path (str): 이미지 파일 경로
            output_base_name (str, optional): 출력 파일 기본 이름, 없으면 이미지 파일 이름 사용
            
        Returns:
            dict: 처리 결과 (텍스트, 시각화 이미지 경로, 텍스트 파일 경로)
        """
        # 기본 출력 이름 설정
        if output_base_name is None:
            output_base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # OCR 실행
        ocr_result, valid_path = self.run_ocr(img_path)
        if ocr_result is None:
            return {"success": False, "message": "인식된 텍스트가 없습니다."}
        
        # 결과 출력
        self.print_ocr_results(ocr_result)
        
        # 시각화 및 저장
        # vis_path = self.visualize_result(valid_path, ocr_result, f'{output_base_name}_result')
        
        # 텍스트 저장
        txt_path = self.save_text_result(ocr_result, f'{output_base_name}_text')
        
        # 추출된 텍스트
        extracted_text = self.get_text_from_result(ocr_result)
        
        return {
            "success": True,
            "text": extracted_text,
            # "visualization_path": vis_path,
            "text_file_path": txt_path,
            "raw_result": ocr_result
        }