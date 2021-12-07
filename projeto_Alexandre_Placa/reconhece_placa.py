import pytesseract
import cv2

def desenhaContornos(contornos, imagem):
    for c in contornos:
        #perimetro do contorno, verifica se o contorno é fechado
        perimetro =cv2.arcLength(c, True)
        if perimetro > 120:
            #aproxima os contornos da forma correspondente
            approx= cv2.approxPolyDP(c, 0.03 * perimetro, True)
            # verifica se é um quadrado ou retângulo de acordo com a qtd de vertisses
            if len(approx) == 4:
                # Contorna a placa atraves dos contornos encontrado
                (x,y, lar, alt) = cv2.boundingRect(c)
                cv2.rectangle(imagem, (x, y), (x + lar, y + alt), (0, 255, 0), 2)
                #segmenta a placa da imagem
                roi = imagem[y:y+alt, x:x+lar]
                cv2.imwrite("rois/roi.png",roi)


def buscaRetanguloPlaca(source):
    #captura ou video
    video = cv2.VideoCapture(source)

    while video.isOpened():

        ret, frame = video.read()

        if(ret == False):
            break


        #area de localizacao u 720p
        area = frame[500:, 300:800]    

        #area de localizacao 480p
        #area = frame[350:, 220:500]

        #escala de cinza
        img_result = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        #limiarizacao
        ret, img_result = cv2.threshold(img_result, (5,5), 0)

        #lista de contorno
        contornos, hier = cv2.findContours(img_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # limite horizontal
        cv2.line(frame, (0, 500), (1280, 500), (0, 0, 255), 1)
        # limite vertical 1
        cv2.line(frame, (300, 0), (300, 720), (0, 0, 255), 1)
        # limite vertical 2
        cv2.line(frame, (800, 0), (800, 720), (0, 0, 255), 1)

        cv2.imshow('FRAME', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    video.release()
    preProcessamentoRoi()
    cv2.destroyAllWindows()


def preProcessamentoRoi():
    
    img_roi= cv2.imread("rois/roi.png")

    if img_roi is None:
        return 

    #rdmensiona a imagem da placa em 4x

    img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation= cv2.INTER_CUBIC)

    #converter para escala de cinza
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #binariza Imagem
    _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)

    # Desfoque na imagem
    img = cv2.GaussianBlur(img, (5,5), 0)

    cv2.imwrite("rois/roi-ocr.png", img)

    return img


def reconhecimentoOCR():
    img_roi_ocr = cv2.imread("rois/roi-ocr.png")

    if img_roi_ocr is None:
        return

    config = r'-c tessedit_char_whitelit=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'     
    saida = pytesseract.image_to_string(img_roi_ocr, lang='eng', config=config)

    print(saida)
    return saida

if __name__ == '__main__':
    
    source = "carro4.jpeg"

    buscaRetanguloPlaca(source)    

    preProcessamentoRoi()

    reconhecimentoOCR()




    
