#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import cv2
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown_udem1')
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

#-----------------------------------------

def nothing(x):
    pass

def yellow(obs):
    lower = np.array([25, 100, 100]) 
    upper = np.array([30, 255, 255])
    mask = cv2.inRange(obs, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    _, img_bin = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
    img_er = cv2.erode(img_bin, kernel, iterations=1)
    img_amarillo = cv2.dilate(img_er, kernel, iterations=3)
    return img_amarillo
def red(obs):
    rojo_bajo = np.array([160, 100, 100])  # Rango bajo de tono (rojo)
    rojo_alto = np.array([180, 255, 255]) # Rango alto de tono (rojo)
    mascara_rojo = cv2.inRange(obs, rojo_bajo, rojo_alto)
    kernel = np.ones((2,2), np.uint8)
    _, img_bin = cv2.threshold(mascara_rojo, 100, 255, cv2.THRESH_BINARY)
    img_er_rojo = cv2.erode(img_bin, kernel, iterations=1)
    img_rojo = cv2.dilate(img_er_rojo, kernel, iterations=3)
    return img_rojo
def white(obs):
    blanco_bajo = np.array([0, 0, 150])  # Rango bajo de tono (blanco)
    blanco_alto = np.array([255, 55, 255]) # Rango alto de tono (blanco)
    mascara_blanco = cv2.inRange(obs, blanco_bajo, blanco_alto)
    kernel = np.ones((5,5), np.uint8)
    _, img_bin = cv2.threshold(mascara_blanco, 100, 255, cv2.THRESH_BINARY)
    img_er = cv2.erode(img_bin, kernel, iterations=2)
    img_blanco = cv2.dilate(img_er, kernel, iterations=4)
    return img_blanco

def hough_canny(imagen, umbral_min, umbral_max,theta=180,rho=1,threshold=100):
    img_canny = cv2.Canny(imagen, umbral_min, umbral_max)
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


cv2.namedWindow('Hough Canny vs Sobel')
def img_t2(obs,longitud_maxima = 1000):
    img_HSV = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
    obs_copy1 = obs.copy()
    obs_copy2 = obs.copy()
    yellow_img = yellow(img_HSV)
    red_img = red(img_HSV)
    white_img = white(img_HSV)
    red_hc, yellow_hc, white_hc = lines_hc(red_img, yellow_img, white_img)
    red_hs, yellow_hs, white_hs = lines_hs(red_img, yellow_img, white_img)
    hc = mostrar_lineas(obs_copy1, red_hc,longitud_maxima,color = (0, 0, 255))
    hc = mostrar_lineas(hc, yellow_hc,longitud_maxima,color = (0, 255, 255))
    hc = mostrar_lineas(hc, white_hc,longitud_maxima,color = (128, 128, 128))
    hs = mostrar_lineas(obs_copy2, red_hs,longitud_maxima,color = (0, 0, 255))
    hs = mostrar_lineas(hs, yellow_hs,longitud_maxima,color = (0, 255, 255))
    hs = mostrar_lineas(hs, white_hs,longitud_maxima,color = (128, 128, 128))
    h_concat = np.concatenate((hc, hs), axis=1)
    cv2.imshow('Hough Canny vs Sobel', h_concat)

#-----------------------------------------
@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    # print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    
    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        #print('done!')
        #env.reset()
        #env.render()
        pass
    
    img_t2(obs,longitud_maxima = 1000)
    cv2.waitKey(1)
    env.render()

pyglet.clock.schedule_interval(update, 0.5 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
