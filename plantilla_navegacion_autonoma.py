#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import math
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Duckietown-udem1-v1")
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

# Parametros para el detector de lineas blancas
white_filter_1 = np.array([0, 0, 0])
white_filter_2 = np.array([0, 0, 0])

# Filtros para el detector de lineas amarillas
yellow_filter_1 = np.array([0, 0, 0])
yellow_filter_2 = np.array([0, 0, 0])
window_filter_name = "filtro"

# Constantes
DUCKIE_MIN_AREA = 0 #editar esto si es necesario
RED_LINE_MIN_AREA = 0 #editar esto si es necesario
RED_COLOR = (0,0,255)
MAX_DELAY = 20
MAX_TOLERANCE = 50

# Variables globales
last_vel = 0.44
delay = -1
tolerance = -1

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render(mode="top_down")


# Funciones interesantes para hacer operaciones interesantes
def box_area(box):
    return abs(box[2][0] - box[0][0]) * abs(box[2][1] - box[0][1])

def bounding_box_height(box):
    return abs(box[2][0] - box[0][0])

def get_angle_degrees2(x1, y1, x2, y2):
    return get_angle_degrees(x1, y1, x2, y2) if y1 < y2 else get_angle_degrees(x2, y2, x1, y1)
    
def get_angle_degrees(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
    if ret_val < 0:
        return 180.0 + ret_val
    return ret_val

def get_angle_radians(x1, y1, x2, y2):
    ret_val = math.atan2(y2 - y1, x2 - x1)
    if ret_val < 0:
        return math.pi + ret_val
    return ret_val


def line_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """ returns a (x, y) tuple or None if there is no intersection """
    d = (by2 - by1) * (ax2 - ax1) - (bx2 - bx1) * (ay2 - ay1)
    if d:
        uA = ((bx2 - bx1) * (ay1 - by1) - (by2 - by1) * (ax1 - bx1)) / d
        uB = ((ax2 - ax1) * (ay1 - by1) - (ay2 - ay1) * (ax1 - bx1)) / d
    else:
        return None, None
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        return None, None
    x = ax1 + uA * (ax2 - ax1)
    y = ay1 + uA * (ay2 - ay1)

    return x, y

def yellow_conds(x1, y1, x2, y2):
    '''
    Condiciones para omitir el procesamiento de una línea amarilla detectada:
    si su ángulo es cercano a 0 o 180, así como si es cercano a recto.
    '''
    angle = get_angle_degrees2(x1, y1, x2, y2)
    return (angle < 30 or angle > 160) or (angle < 110 and angle > 90)

def white_conds(x1, y1, x2, y2):
    '''
    Condiciones para omitir el procesamiento de una línea blanca detectada:
    si se encuentra en el primer, segundo o tercer cuadrante, en otras palabras,
    se retorna False solo si la línea está en el cuarto cuadrante.
    '''
    return (min(x1,x2) < 320) or (min(y1,y2) < 320)



def duckie_detection(obs, converted, frame):
    '''
    Detectar patos, retornar si hubo detección y el ángulo de giro en tal caso 
    para lograr esquivar el duckie y evitar la colisión.
    '''

    # Se asume que no hay detección
    detection = False
    angle = 0

    '''
    Para lograr la detección, se puede utilizar lo realizado en el desafío 1
    con el freno de emergencia, aunque con la diferencia que ya no será un freno,
    sino que será un método creado por ustedes para lograr esquivar al duckie.
    '''

    # Implementar filtros


    # Realizar la segmentación con operaciones morfológicas (erode y dilate) 
    # y buscar los contornos


    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    # a la detección, además, dentro de este for, se establece la detección = verdadera
    # además del ángulo de giro angle = 'ángulo'

    # Mostrar ventanas con los resultados
    #cv2.imshow("Patos filtro", segment_image_post_opening)
    #cv2.imshow("Patos detecciones", frame)

    return detection, angle



def red_line_detection(converted, frame):
    '''
    Detección de líneas rojas en el camino, esto es análogo a la detección de duckies,
    pero con otros filtros, notar también, que no es necesario aplicar houghlines en este caso
    '''
    # Se asume que no hay detección
    detection = False


    # Implementar filtros


    # Realizar la segmentación con operaciones morfológicas (erode y dilate) 
    # y buscar los contornos

 

    # Revisar los contornos identificados y dibujar el rectángulo correspondiente
    # a la detección, además, Si hay detección, detection = True
    


    # Mostrar ventanas con los resultados

    return detection



def get_line(converted, filter_1, filter_2, line_color):
    '''
    Determina el ángulo al que debe girar el duckiebot dependiendo
    del filtro aplicado, y de qué color trata, si es "white"
    y se cumplen las condiciones entonces gira a la izquierda,
    si es "yellow" y se cumplen las condiciones girar a la derecha.
    '''
    mask = cv2.inRange(converted, filter_1, filter_2)
    segment_image = cv2.bitwise_and(converted, converted, mask=mask)
    
    # Erosionar la imagen
    image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)
    kernel = np.ones((5,5),np.uint8)
    image_lines = cv2.erode(image, kernel, iterations = 2)    

    # Detectar líneas
    gray_lines = cv2.cvtColor(image_lines, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_lines, 50, 200, None, 3)
   
    # Detectar lineas usando houghlines y lo aprendido en el desafío 2.


    cv2.imshow(line_color, image_lines)

    # Se cubre cada color por separado, tanto el amarillo como el blanco
    # Con esto, ya se puede determinar mediante condiciones el movimiento del giro del robot.
    # Por ejemplo, si tiene una linea blanca muy cercana a la derecha, debe doblar hacia la izquierda
    # y viceversa.
    return 0, 0

def line_follower(vel, angle, obs):
    converted = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
    frame = obs[:, :, [2, 1, 0]]
    frame = cv2.UMat(frame).get()

    # Detección de duckies
    detection, _ = duckie_detection(obs=obs, frame=frame, converted=converted)
    '''
    Implementar evasión de duckies en el camino, variado la velocidad angular del robot
    '''


    # Detección de líneas rojas
    detection = red_line_detection(converted=converted, frame=frame)
    ''' 
    Implementar detención por un tiempo determinado del duckiebot
    al detectar una linea roja en el camino, luego de este tiempo,
    el duckiebot debe seguir avanzando
    '''


    # Obtener el ángulo propuesto por cada color
    _, angle_white = get_line(converted, white_filter_1, white_filter_2, "white")
    _, angle_yellow = get_line(converted, yellow_filter_1, yellow_filter_2, "yellow")


    '''
    Implementar un controlador para poder navegar dentro del mapa con los ángulos obtenidos
    en las líneas anteriores
    '''

    return np.array([vel, 'new_angle']) # Implementar nuevo ángulo de giro controlado

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    Handler para reiniciar el ambiente
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render(mode="top_down")

    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0

    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)


# Registrar el handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

action = np.array([0.44, 0.0])

def update(dt):
    """
    Funcion que se llama en step.
    """
    global action
    # Aquí se controla el duckiebot
    if key_handler[key.UP]:
        action[0] += 0.44
    if key_handler[key.DOWN]:
        action[0] -= 0.44
    if key_handler[key.LEFT]:
        action[1] += 1
    if key_handler[key.RIGHT]:
        action[1] -= 1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    ''' Aquí se obtienen las observaciones y se setea la acción
    Para esto, se debe utilizar la función creada anteriormente llamada line_follower,
    la cual recibe como argumentos la velocidad lineal, la velocidad angular y 
    la ventana de la visualización, en este caso obs.
    Luego, se setea la acción del movimiento implementado con el controlador
    con action[i], donde i es 0 y 1, (lineal y angular)
    '''


    # obs consiste en un imagen de 640 x 480 x 3
    obs, reward, done, info = env.step(action)
    vel, angle = line_follower(action[0], action[1], obs)
    action[0] = vel
    action[1] = angle

    if done:
        print('done!')
        env.reset()
        env.render(mode="top_down")

    env.render(mode="top_down")

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()