import os
from PIL import Image
import pytesseract

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Point pytesseract at the installed Tesseract engine
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = r"D:\OCR-TESSERACT\tesseract.exe"

# Change this to the image you want to OCR
image_path = "ocr-sample2.jpg"

image = Image.open(os.path.join(SCRIPT_DIR, image_path))
text = pytesseract.image_to_string(image)

print(f"OCR result for {image_path}:")
print(text)
