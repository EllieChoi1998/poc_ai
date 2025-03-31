import os
from pdf2image import convert_from_path

class PDFtoPNG:
    def __init__(self, source_dir, save_dir):
        self.source_dir = source_dir
        self.save_dir = save_dir

    def convert_one_pdf(self, filename):
        images = convert_from_path(os.path.join(self.source_dir, filename))
        pdf_name = filename.split(".")
        converted_path = []
        for i, image in enumerate(images):
            image.save(os.path.join(self.save_dir, f'{pdf_name[0]}_{i}.png'), 'PNG')
            converted_path.append(f"{self.save_dir}/{pdf_name[0]}_{i}.png")
        return converted_path

    def convert_all(self):
        for filename in os.listdir(self.source_dir):
            if filename.endswith(".pdf"):  # PDF 파일만 처리
                images = convert_from_path(os.path.join(self.source_dir, filename))
                pdf_name = filename.split(".")
                for i, image in enumerate(images):
                    image.save(os.path.join(self.save_dir, f'{pdf_name[0]}_{i}.png'), 'PNG')
