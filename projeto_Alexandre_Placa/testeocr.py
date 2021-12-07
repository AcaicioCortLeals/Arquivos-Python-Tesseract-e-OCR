import cv2
import pytesseract


#incorporar imagem
img = cv2.imread("xerox.png") 

caminho = r'C:\Users\aa\AppData\Local\Programs\Tesseract-OCR'

valor = r'\tesseract.exe'


pytesseract.pytesseract.tesseract_cmd = caminho+valor

texto = pytesseract.image_to_string(img)

print(texto)
cv2.imshow("imagem", img)

cv2.waitKey(0)
cv2.destroyAllWindows()








