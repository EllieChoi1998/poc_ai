# Import required libraries
from paddlex import create_model
import cv2
import os
import numpy as np
from bs4 import BeautifulSoup
from ocrEngine import PaddleEngine  # Import your PaddleEngine class

def main():
    # Initialize models and paths
    image_path = "./input/pdf_5_1_0.png"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Initialize models
    print("Loading document layout model...")
    model_name = "PP-DocLayout-L"
    layout_model = create_model(model_name=model_name)
    
    print("Loading table recognition model...")
    table_model = create_model(model_name="SLANet")
    
    # Initialize PaddleOCR engine for text detection in cells
    print("Initializing PaddleOCR engine...")
    ocr_engine = PaddleEngine(use_gpu=True, lang="korean")
    
    # Step 1: Document layout analysis
    print(f"Performing layout detection on {image_path}...")
    output = layout_model.predict(image_path, batch_size=1, layout_nms=True)
    
    # Extract table positions
    table_positions = []
    for res in output:
        table_count = 0
        box_count = 0
        for box in res['boxes']:
            if box['label'] == 'table':
                table_positions.append(box['coordinate'])
                table_count += 1
            box_count += 1
        print(f"Found {box_count} elements, including {table_count} tables")
    
    if not table_positions:
        print("No tables detected in the image")
        return
    
    # Step 2: Crop and save detected tables
    saved_tables = []
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for idx, (x1, y1, x2, y2) in enumerate(table_positions):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers
        cropped_table = image[y1:y2, x1:x2]  # Crop the image
        
        # Save the cropped table
        save_path = f"{output_dir}/{image_name}_table_{idx+1}.png"
        saved_tables.append((save_path, (x1, y1, x2, y2)))  # Store path and coordinates
        cv2.imwrite(save_path, cropped_table)
        print(f"Saved table {idx+1} at {save_path}")
    
    # Step 3: Table structure recognition and cell text detection
    final_results = []
    
    for idx, (table_img_path, table_coords) in enumerate(saved_tables):
        table_base_x, table_base_y = table_coords[0], table_coords[1]
        print(f"\nProcessing table {idx+1}: {table_img_path}")
        
        # Recognize table structure
        table_output = table_model.predict(input=table_img_path, batch_size=1)
        
        for res in table_output:
            # Parse HTML structure
            html_structure = ''.join(res['structure'])
            soup = BeautifulSoup(html_structure, 'html.parser')
            
            # Check if we have cell bounding boxes
            if len(res['bbox']) > 0:
                print(f"Found {len(res['bbox'])} cells in the table")
                
                # Initialize list to store OCR results for each cell
                ocr_texts = []
                
                # Original table image for cropping cells
                table_img = cv2.imread(table_img_path)
                if table_img is None:
                    print(f"Error: Could not load table image at {table_img_path}")
                    continue
                
                # Process each cell
                for cell_idx, bbox in enumerate(res['bbox']):
                    x_values = bbox[::2]  # x coordinates
                    y_values = bbox[1::2]  # y coordinates
                    
                    x_min, x_max = int(min(x_values)), int(max(x_values))
                    y_min, y_max = int(min(y_values)), int(max(y_values))
                    
                    # Add padding to ensure text is fully captured
                    padding = 2
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(table_img.shape[1], x_max + padding)
                    y_max = min(table_img.shape[0], y_max + padding)
                    
                    # Crop the cell image
                    cell_img = table_img[y_min:y_max, x_min:x_max]
                    
                    if cell_img.size == 0:
                        print(f"Warning: Empty cell image for cell {cell_idx+1}")
                        ocr_texts.append("")
                        continue
                    
                    # Save cell image for debugging (optional)
                    cell_img_path = f"{output_dir}/{image_name}_table_{idx+1}_cell_{cell_idx+1}.png"
                    cv2.imwrite(cell_img_path, cell_img)
                    
                    # Perform OCR on the cell using PaddleEngine
                    try:
                        ocr_result, _ = ocr_engine.run_ocr(cell_img_path)
                        if ocr_result and len(ocr_result) > 0:
                            cell_text = ocr_engine.get_text_from_result(ocr_result)
                            ocr_texts.append(cell_text)
                        else:
                            ocr_texts.append("")
                    except Exception as e:
                        print(f"Error performing OCR on cell {cell_idx+1}: {e}")
                        ocr_texts.append("")
                
                # Update HTML table with OCR results
                table_cells = soup.find_all("td")
                for i, cell in enumerate(table_cells):
                    if i < len(ocr_texts):
                        cell.string = ocr_texts[i]  # Insert OCR result
                
                # Save final HTML
                final_html = soup.prettify()
                html_output_path = f"{output_dir}/{image_name}_table_{idx+1}_result.html"
                with open(html_output_path, "w", encoding="utf-8") as f:
                    f.write(final_html)
                
                print(f"Saved HTML result to {html_output_path}")
                final_results.append({
                    "table_index": idx + 1,
                    "table_coords": table_coords,
                    "html_path": html_output_path,
                    "html_content": final_html
                })
            else:
                print("No cells found in the table structure")
    
    print("\nProcessing complete!")
    print(f"Processed {len(final_results)} tables")
    
    # Return results for further use if needed
    return final_results

if __name__ == "__main__":
    main()