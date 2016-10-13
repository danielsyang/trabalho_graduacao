import numpy as np
import sys
import cv2


def main():
    path = '/home/shu/Desktop/teste/testeFinal/'  # lembrar de colocar '/' no final do path
    nome_imagem = 'P_20160919_220028.jpg'
    title_imagem = nome_imagem[:-4]
    kernel = np.ones((1, 1), np.uint8)
    im = cv2.imread(path + nome_imagem)
    height, width, channels = im.shape

    r_size = int(width / 500 + 1)
    # heigth_size = int(height / 500 + 1)

    im_size = cv2.resize(im, (width / r_size, height / r_size), cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(im_size, cv2.COLOR_BGR2GRAY)
    height, width = img_gray.shape

    size_se = int(height / 50)
    if size_se % 2 == 0:
        size_se += 1

    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_median_blur = cv2.medianBlur(img_blur, 5)

    thresh = cv2.adaptiveThreshold(img_median_blur, 255, 1, cv2.THRESH_BINARY, 11, 2)

    thresh_copia = cv2.adaptiveThreshold(img_median_blur, 255, 1, cv2.THRESH_BINARY, 11, 2)

    #      Achar os contornos
    not_used, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imagem = 10
    for cnt in contours:

        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > 15:
            cv2.rectangle(im_size, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.imshow('Imagem', im_size)
            key = cv2.waitKey(0)
            if key == 27:
                sys.exit()
            elif key == 10:
                cropped = thresh_copia[y - 5:y + h + 5, x - 5:x + w + 5]
                resized_image = cv2.resize(cropped, (28, 28))
                negative_resized_image = negativo_imagem(resized_image)
                opening = cv2.morphologyEx(negative_resized_image, cv2.MORPH_OPEN, kernel)
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
                cv2.imwrite("/home/shu/Desktop/teste/testeFinal/" + title_imagem + "_" + str(imagem) + ".png", closing)
                imagem += 1


def negativo_imagem(img):
    return np.subtract(255, img)


if __name__ == '__main__':
    main()
