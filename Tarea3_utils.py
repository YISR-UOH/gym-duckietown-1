import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import euclidean
from heapq import heappop, heappush
"""
  Detector de lineas usando sobel y canny, ajustes para vista aeria
"""
def canny(imagen, umbral_min=50, umbral_max=150, kernel=3, sigma=0):
    gauss = cv2.GaussianBlur(imagen, (kernel, kernel), sigma)
    canny = cv2.Canny(gauss, umbral_min, umbral_max)
    return canny
def sobel(imagen, umbral_min=50, umbral_max=150, kernel=3, sigma=0):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(imagen, (kernel, kernel), sigma)
    sobel_x = cv2.Sobel(gauss, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gauss, cv2.CV_64F, 0, 1, ksize=3)
    gradiente_magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
    return gradiente_magnitud
def custom_bordes(imagen):
  bordes = sobel(imagen, kernel=7, sigma=2)
  bordes = cv2.convertScaleAbs(bordes)
  bordes = canny(bordes, 100, 500, 5, 4)
  kernel = np.ones((7,7), np.uint8)
  _, img_bin = cv2.threshold(bordes, 100, 255, cv2.THRESH_BINARY)
  b = cv2.dilate(img_bin, kernel, iterations=2)
  return b

def get_areas(matriz):
  try:
    # Etiquetar áreas usando connectedComponentsWithStats
    matriz = np.array(matriz, dtype=np.uint8)
    # Invertir la matriz para que las áreas de interés sean 1
    matriz_invertida = np.where(matriz == 0, 1, 0).astype(np.uint8)

    # Etiquetar áreas usando connectedComponentsWithStats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(matriz_invertida, connectivity=8)

    
    areas = stats[:, cv2.CC_STAT_AREA]
    areas_con_etiquetas = [(label, areas[label]) for label in range(1, num_labels)]
    areas_con_etiquetas.sort(key=lambda x: x[1], reverse=True)
    labeled_img = np.where(labels == 2, 255, 0).astype(np.uint8)
    _, binary_image = cv2.threshold(labeled_img , 127, 1, cv2.THRESH_BINARY)
    inverted_binary_image = 1 - binary_image
    num_labels, labels_im = cv2.connectedComponents(inverted_binary_image)
    min_size = 500
    output_image = np.zeros_like(binary_image)
    for label in range(1, num_labels):
      component = (labels_im == label)
      if np.sum(component) >= min_size:
        output_image[component] = 1
    kernel = np.ones((3, 3), np.uint8)
    labeled_img_fix = cv2.morphologyEx(output_image, cv2.MORPH_CLOSE, kernel)
        
    return areas_con_etiquetas, matriz, labels,labeled_img_fix
  
  except Exception as e:
    print(e)
    return (None,None,None)
    

def check_neighbors(matrix, x, y, d):
  rows, cols = matrix.shape
  for i in range(max(0, x-d), min(rows, x+d+1)):
      for j in range(max(0, y-d), min(cols, y+d+1)):
          if (i, j) != (x, y):  # Exclude the center point itself
              if matrix[i, j] == 1:
                return False
  return True
    
def get_Bpoint(labeled_img):
    if labeled_img is None:
        return (0,0)
    points = np.column_stack(np.where(labeled_img == 0))
    while True:
        point = random.choice(points)
        # check if the point is at least d distance from any border 
        if labeled_img[point[0],point[1]] == 0:
          if check_neighbors(labeled_img, point[0], point[1], 20):
            break
    return point
    
def heuristic(a, b):
    """Heurística de distancia Manhattan"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(matrix, start, goal):
    """Implementación del algoritmo A*"""
    rows, cols = matrix.shape
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heappop(open_set)[1]

        if current == goal:
            # Reconstruir el camino
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and matrix[neighbor[1],neighbor[0]] == 1:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No se encontró un camino
  
def get_duckiePosition(obs):
  try:
    img_HSV = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
    rojo_bajo = np.array([100, 180, 120])  # Rango bajo de tono (rojo)
    rojo_alto = np.array([120, 255,180]) # Rango alto de tono (rojo)
    mascara_rojo = cv2.inRange(img_HSV, rojo_bajo, rojo_alto)
    kernel = np.ones((2,2), np.uint8)
    _, img_bin = cv2.threshold(mascara_rojo, 100, 255, cv2.THRESH_BINARY)
    img_er_rojo = cv2.erode(img_bin, kernel, iterations=2)
    kernel = np.ones((5,5), np.uint8)
    duckie = cv2.dilate(img_er_rojo, kernel, iterations=4)
    _, labels, _, centroids = cv2.connectedComponentsWithStats(duckie, connectivity=8)
    # Cambiar el rango de etiquetas para visualizar mejor (opcional)
    label_hue = np.uint8(190 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Convertir de HSV a BGR para mostrar con matplotlib
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # Configurar el color de fondo (etiqueta 0)
    labeled_img[label_hue == 0] = 0
    return labeled_img,centroids[1]
  except:
    return None,None
  
def PathPlanning(matriz,posicion_robot,punto_b):
  try:
    inicio = (int(posicion_robot[0]),int(posicion_robot[1]))
    punto_b = (punto_b[1],punto_b[0])
    camino = a_star(matriz, inicio, punto_b)
    if camino is None:
      return None
    if len(camino) <= 1:
      #print("Error: No se puede planificar el camino")
      return camino    
    return camino
  except Exception as e:
    return None
  
