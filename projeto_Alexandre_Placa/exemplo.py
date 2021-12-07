import cv2
import pytesseract

#padrão 90 e 70 respectivamente

def encontrarRoiPlaca(source):
    img = cv2.imread(source)
    cv2.imshow("img", img)

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("cinza", img)

    _, bin = cv2.threshold(cinza, 90, 255, cv2.THRESH_BINARY)
    # cv2.imshow("binary", img)

    desfoque = cv2.GaussianBlur(bin, (5, 5), 0)
    # cv2.imshow("defoque", desfoque)

    contornos, hierarquia = cv2.findContours(desfoque, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(img, contornos, -1, (0, 255, 0), 1)

    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        if perimetro >200 :
            aprox = cv2.approxPolyDP(c, 0.03 * perimetro, True)
            if len(aprox) == 4:
                (x, y, alt, lar) = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + alt, y + lar), (0, 255, 0), 2)
                roi = img[y:y + lar, x:x + alt]
                cv2.imwrite('rois/roi.png', roi)

    cv2.imshow("contornos", img)


def preProcessamentoRoiPlaca():
    img_roi = cv2.imread("rois/roi.png")

    if img_roi is None:
        return

    resize_img_roi = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Converte para escala de cinza
    img_cinza = cv2.cvtColor(resize_img_roi, cv2.COLOR_BGR2GRAY)

    # Binariza imagem
    _, img_binary = cv2.threshold(img_cinza, 70, 255, cv2.THRESH_BINARY)

    
    # Desfoque na Imagem
    img_desfoque = cv2.GaussianBlur(img_binary, (5, 5), 0)

    # Grava o pre-processamento para o OCR
    cv2.imwrite("rois/roi-ocr.png", img_desfoque)

    cv2.imshow("ROI", img_desfoque)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_desfoque


def ocrImageRoiPlaca():
    image = cv2.imread("rois/roi-ocr.png")

    config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

    caminho = r'C:\Users\aa\AppData\Local\Programs\Tesseract-OCR'

    valor = r'\tesseract.exe'

    pytesseract.pytesseract.tesseract_cmd = caminho+valor 


    saida = pytesseract.image_to_string(image, lang='eng', config=config)

    return saida


if __name__ == "__main__":

    source = "carro4.jpg"
    encontrarRoiPlaca(source)

    pre = preProcessamentoRoiPlaca()

    ocr = ocrImageRoiPlaca()

    placa = ocr.split()


    
    if ocr == "":
        print("Erro ao ler a imagem")
    
    else:
        cv2.imshow("placa", cv2.imread("rois/roi.png"))
        
        #escrever no terminal o valor da placa
        for valor in placa:
            if(len(valor)>=7):
                print(valor) 

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    