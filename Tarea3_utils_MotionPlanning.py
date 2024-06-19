import cv2
import numpy as np
import math


def yellow(obs):
    lower = np.array([25, 100, 100]) 
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(obs, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    _, img_bin = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    img_er = cv2.erode(img_bin, kernel, iterations=1)
    img_amarillo = cv2.dilate(img_er, kernel, iterations=4)
    return img_amarillo
  
def red(obs):
    rojo_bajo = np.array([160, 100, 100])  # Rango bajo de tono (rojo)
    rojo_alto = np.array([180, 255, 255]) # Rango alto de tono (rojo)
    mascara_rojo = cv2.inRange(obs, rojo_bajo, rojo_alto)
    kernel = np.ones((2,2), np.uint8)
    _, img_bin = cv2.threshold(mascara_rojo, 100, 255, cv2.THRESH_BINARY)
    img_er_rojo = cv2.erode(img_bin, kernel, iterations=1)
    img_rojo = cv2.dilate(img_er_rojo, kernel, iterations=4)
    return img_rojo
  
def white(obs):
    blanco_bajo = np.array([0, 0, 150])  # Rango bajo de tono (blanco)
    blanco_alto = np.array([255, 55, 255]) # Rango alto de tono (blanco)
    mascara_blanco = cv2.inRange(obs, blanco_bajo, blanco_alto)
    kernel = np.ones((5,5), np.uint8)
    _, img_bin = cv2.threshold(mascara_blanco, 100, 255, cv2.THRESH_BINARY)
    img_er = cv2.erode(img_bin, kernel, iterations=2)
    img_blanco = cv2.dilate(img_er, kernel, iterations=5)
    return img_blanco

def hough_canny(imagen, umbral_min, umbral_max,theta=180,rho=1,threshold=100):
    img_canny = cv2.Canny(imagen, umbral_min, umbral_max,apertureSize=3)
    lines_canny = cv2.HoughLines(img_canny,  rho, np.pi / theta, threshold)
    return lines_canny

def hough_sobel(imagen, umbral_min, umbral_max,theta=180,rho=1,threshold=100):
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=5)
    gradiente_magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
    img_sobel = cv2.convertScaleAbs(gradiente_magnitud)
    bordes = cv2.Canny(img_sobel, umbral_min, umbral_max)
    lines_sobel = cv2.HoughLines(bordes, rho, np.pi / theta, threshold)
    return lines_sobel

def mostrar_lineas(imagen, lines,longitud_maxima = 1000,color = (0, 255, 255)):
    if lines is not None:
      for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + longitud_maxima * (-b))
            y1 = int(y0 + longitud_maxima * (a))
            x2 = int(x0 - longitud_maxima * (-b))
            y2 = int(y0 - longitud_maxima * (a))
            cv2.line(imagen, (x1, y1), (x2, y2), color, 2)
      return imagen
    return imagen

def mostrar_lineas2(imagen, lines,color ,flag):
    if lines is None:
        return imagen
    for line in lines:
        # considerar los puntos que esten debajo del 50% de la imagen
        for x1, y1, x2, y2 in line:
            if x1 > imagen.shape[0] * 0.6 and x2 > imagen.shape[0] * 0.6:
                continue
            if y1 < imagen.shape[1] * 0.3 and y2 < imagen.shape[1] * 0.3:
                continue
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Clasificar la línea basada en su ángulo
            if abs(angle) < 10 and flag=="red":  # Línea horizontal
                cv2.line(imagen, (x1, y1), (x2, y2), color, 2)
            elif 10 <= abs(angle) < 80 and flag=="vias":  # Línea levemente inclinada
                cv2.line(imagen, (x1, y1), (x2, y2), color, 2)
            elif 70 <= abs(angle) <= 100 and flag!="red":  # Línea vertical
                cv2.line(imagen, (x1, y1), (x2, y2), color, 2)
    return imagen

  
def lines_hs(red, yellow, white):
    red_hs = hough_sobel(red, 50, 150,theta=100,rho=1.5,threshold=120)
    yellow_hs = hough_sobel(yellow,50, 150,theta=160,rho=1.9,threshold=100)
    white_hs = hough_sobel(white, 50, 150,theta=100,rho=1.5,threshold=120)
    return red_hs, yellow_hs, white_hs
  
def lines_hc(red, yellow, white):
    red_hc = hough_canny(red, 50, 150,theta=100,rho=1.5,threshold=120)
    yellow_hc = hough_canny(yellow,50, 150,theta=120,rho=3,threshold=100)
    white_hc = hough_canny(white,50, 150,theta=180,rho=1.8,threshold=120)
    return red_hc, yellow_hc, white_hc
