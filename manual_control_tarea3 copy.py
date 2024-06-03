#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
import pyglet
from pyglet.gl import *
import numpy as np
from PIL import Image
import cv2
from pyglet.window import key
import numpy as np
from PIL import Image
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import time
import Tarea3_utils as tarea3
import Tarea3_utils_PathPlanning as tpp

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown_udem1')
parser.add_argument('--map-name', default='proyecto_final')
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
data = []
for i in env.grid:
    data.append([i["coords"], i["kind"], i["drivable"]])

matriz,data = tpp.data_map(data)
matrix_color = np.zeros((matriz.shape[0], matriz.shape[1], 3), dtype=np.uint8)
# Asignar colores: negro para 0 y blanco para 1
matrix_color[matriz == 0] = [0, 0, 0]
matrix_color[matriz == 1] = [255, 255, 255]
# el valor 1 es para las posiciones donde se puede conducir
PointB = tpp.randomPoint(matriz)
cv2.namedWindow("Path_Planning", cv2.WINDOW_NORMAL)
cv2.namedWindow("top", cv2.WINDOW_NORMAL)


def get_pos():
    n = 2
    road_tile_size = env.road_tile_size
    x,y = env.cur_pos[0],env.cur_pos[2]
    x0 = x//road_tile_size # posicion en x en la matriz original
    y0 = y//road_tile_size # posicion en y en la matriz original
    x1 = x-x0*road_tile_size # posicion en x en la matriz de n
    y1 = y-y0*road_tile_size # posicion en y en la matriz de n
    if x1 >= road_tile_size/2:
        x1 = 1
    else:
        x1 = 0
    if y1 >= road_tile_size/2:
        y1 = 1
    else:
        y1 = 0
    return (int(y0*n+y1),int(x0*n+x1))
PointA = get_pos()
path,_ = tpp.Path(data,PointA,PointB)
def update_Path(dt):
    global matrix
    global data
    global path
    global PointA
    global PointB
    PointA = get_pos()
    path,_ = tpp.Path(data,PointA,PointB)

def Path_planning():
    global matriz
    global matrix_color
    global data
    global PointA,PointB
    
    copia = matrix_color.copy()
    width, height = env.unwrapped.window.get_size()
    buffer = (GLubyte * (3 * width * height))(0)
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer)
    image_data = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
    image_data = np.flipud(image_data)

    
    
    if path is not None:
        for i in path:
            x,y = data.loc[data['node'] == i, 'pos'].values[0]
            copia[x,y] = (150, 0, 0)
    
    copia[PointB] = (250, 0, 255)
    copia[PointA] = (250, 0, 0)
    cv2.imshow("Path_Planning", copia)

    
    cpia2 = image_data.copy()
    cv2.imshow("top", cpia2)

cv2.namedWindow("Motion_Planning", cv2.WINDOW_NORMAL)
cv2.namedWindow("Líneas Amarillas", cv2.WINDOW_NORMAL)
cv2.namedWindow("Líneas Rojas", cv2.WINDOW_NORMAL)
cv2.namedWindow("Orilla Gris", cv2.WINDOW_NORMAL)
def Motion_Planning(obs):
    import Tarea3_utils_MotionPlanning as tmp
    
    hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
    vias = tmp.yellow(hsv)
    stop = tmp.red(hsv)
    vereda = tmp.white(hsv) 
    
    if vias is not None:
        cv2.imshow('Líneas Amarillas', vias)
    if stop is not None:
        cv2.imshow('Líneas Rojas', stop)
    if vereda is not None:  
        cv2.imshow('Orilla Gris', vereda)
    cv2.imshow("Motion_Planning", obs)
    
    return None
    
#-----------------------------------------


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """
    global PointA,PointB

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        env.reset()
        env.render()
        print('RESET')
        PointB = tpp.randomPoint(matriz)
        PointA = get_pos()
        print("nuevo destino: ",PointB)
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        print(PointA,PointB)
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
    global flag
    
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

    if done:
        #print('done!')
        #env.reset()
        #env.render()
        pass
    
    # print(env.cur_pos) [3.83193618 0.         2.66608419]
    # print(env.cur_angle) 3.1352243456171034
    # print(env.grid_width) 17 
    # print(env.grid_height) 9
    # print(env.road_tile_size) 0.585

 
    
    cv2.waitKey(1)
    env.render(mode="top_down")
    Path_planning()
    Motion_Planning(obs)
    

pyglet.clock.schedule_interval(update, 0.5 / env.unwrapped.frame_rate)
pyglet.clock.schedule_interval(update_Path, 10 / env.unwrapped.frame_rate)
# Enter main event loop
pyglet.app.run()

env.close()
