import os

print(os.getcwd())

import easyocr

reader = easyocr.Reader(['en'])

#result = reader.readtext("ocrimage1.jpg")

#result = reader.readtext("ocrimageindiamap1.jpeg")
result = reader.readtext("ocrimageeyetest2.jpeg")


for r in result:
    print(r[1])