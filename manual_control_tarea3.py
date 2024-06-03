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
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import time
import Tarea3_utils as tarea3
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
# Create a top-down environment for the map
cv2.namedWindow("normal", cv2.WINDOW_NORMAL)
cv2.namedWindow("top", cv2.WINDOW_NORMAL)
cv2.namedWindow("top3", cv2.WINDOW_NORMAL)
areas_con_etiquetas, matriz, labels = None,None,None
punto_b = (0,0)
b = None
labeled_img = None
flag = 0
copia = None
def test(time=0):
    global areas_con_etiquetas, matriz, labels
    global punto_b
    global b
    global labeled_img
    global flag
    global copia
    width, height = env.unwrapped.window.get_size()
    buffer = (GLubyte * (3 * width * height))(0)
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer)
    image_data = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
    image_data = np.flipud(image_data)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    if flag == 1:
        cv2.imshow("top", image_data)
        im = Image.fromarray(image_data)
        im.save('top'+str(time)+'.png')
        flag = 0
    cv2.imshow("top", image_data)
    if b is None:
        b = tarea3.custom_bordes(image_data)
        areas_con_etiquetas, matriz, labels,labeled_img = tarea3.get_areas(b)
    if punto_b[0] == 0 and punto_b[1] == 0: 
        punto_b = tarea3.get_Bpoint(labeled_img)
        
    duckie,position = tarea3.get_duckiePosition(image_data)
    if labeled_img is not None and position is not None and punto_b is not None:
        copia_ruta = np.copy(image_data)
        copia_ruta_b = np.copy(labeled_img)
        ruta = tarea3.PathPlanning(copia_ruta_b,position,punto_b)
        # colocar la ruta en labeled_img en verde
        cv2.circle(copia_ruta, (punto_b[0],punto_b[1]), 10, (255,0,0), -1)
        cv2.circle(copia_ruta, (int(position[0]),int(position[1])), 10, (255,0,0), -1)
        cv2.circle(copia_ruta, (punto_b[1],punto_b[0]), 10, (255,100,0), -1)
        cv2.circle(copia_ruta, (500,500), 50, (255,0,0), -1)
        cv2.circle(copia_ruta, (500,300), 10, (255,0,0), -1)
        cv2.circle(copia_ruta, (300,500), 30, (255,0,0), -1)
        cv2.imshow("top3", copia_ruta)
        if ruta is not None:
            for p in ruta:
                cv2.circle(copia_ruta, p, 5, (100, 255, 0), -1)
            cv2.imshow("top3", copia_ruta)
        
    
        
    
#-----------------------------------------


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    global areas_con_etiquetas, matriz, labels
    global punto_b
    global b
    global labeled_img
    global flag
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        env.reset()
        env.render()
        punto_b = (0,0)
        flag = 0
        # reset windows
        cv2.destroyAllWindows()
        time.sleep(1)
        cv2.namedWindow("normal", cv2.WINDOW_NORMAL)
        cv2.namedWindow("top", cv2.WINDOW_NORMAL)
        cv2.namedWindow("top3", cv2.WINDOW_NORMAL)
        print('RESET')
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
    
    
    cv2.waitKey(1)
    env.render(mode="top_down")
    print(env.grid)
    cv2.imshow("normal", obs)
    test()
    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)
        t = time.time()
        im.save('normal'+str(t)+'.png')
        flag = 1
        test(time=t)

pyglet.clock.schedule_interval(update, 1.5 / env.unwrapped.frame_rate)
# Enter main event loop
pyglet.app.run()

env.close()
