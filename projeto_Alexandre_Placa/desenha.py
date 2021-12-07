#importar o opencv
import cv2
import pytesseract







def encontrarRoiPlaca(source):
    #importar imagenm
    img = cv2.imread(source)

    #cv2.imshow("imagem", img)

    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    #transformar em uma imagem limiarizada (limte para branco e preto)
    _, thresh = cv2.threshold(cinza, 90, 255, cv2.THRESH_BINARY)

    #cv2.imshow("cinza", cinza)
    # cv2.imshow("thresh", thresh)

    #aplicar desfoue
    desfoque = cv2.GaussianBlur(thresh, (5,5), 0)


    #procurar contorno
    contorno, hier = cv2.findContours(desfoque, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )

    #desenhar contornos
    #cv2.drawContours(img, contorno, -1, (0,255,0), 2 )

    for c in contorno:
        perimetro = cv2.arcLength(c, True)

        if(perimetro>120):

            aprox = cv2.approxPolyDP(c, 0.03*perimetro, True)

            if len(aprox) == 4:
                (x, y, h, w) = cv2.boundingRect(c)
                cv2.rectangle(img, (x,y), (x+h, y+w), (0,255,0), 2)
                roi = img[y:y+w, x:x+h]
                #roi = cv2.resize(roi , (400,200))
                cv2.imwrite('rois/roi.jpg', roi)
            

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    source = 'carro4.jpg'

    encontrarRoiPlaca(source)

    caminho = r'C:\Users\aa\AppData\Local\Programs\Tesseract-OCR'

    valor = r'\tesseract.exe'

    pytesseract.pytesseract.tesseract_cmd = caminho+valor 

    #imagem =  cv2.resize('./rois/roi.jpg', None, 4, 4, interpolation=cv2.INTER_CUBIC)
    imagem =  cv2.imread('./rois/roi.jpg')
    imagem =  cv2.resize(imagem, (500,175) , interpolation=cv2.INTER_CUBIC)

    


    texto = pytesseract.image_to_string(imagem)

    print(texto)