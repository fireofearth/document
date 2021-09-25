
# Built-in libraries
import os
import sys
from glob import glob
import random
import time

# PyPI libraries
import numpy as np
import cv2
from PIL import Image

# Local libararies
import carla
import utility as util
import carlautil

IMG_WIDTH = 640
IMG_HEIGHT = 480
SHOW_PREVIEW = False
SHOULD_SAVE_EACH_IMG = False

os.makedirs('figs', exist_ok=True)

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0

    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    actor_list = []

    front_camera = None
    collision_hist = []

def process_img(image):
    print("In process_img, frame", image.frame)
    img = np.array(image.raw_data)
    img = img.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    img = img[:, :, :3]
    if SHOULD_SAVE_EACH_IMG:
        cv2.imwrite(f"figs/snapshot-{image.frame}.png", img)
        # _img = img.astype('uint8')
        # pil_img = Image.fromarray(_img, mode='RGB')
        # pil_img.save(f"figs/snapshot-{image.frame}.png")
    return img / 255.0

def main():
    vehicles = []
    sensors = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('model3')[0]
        # spawn_point = random.choice(world.get_map().get_spawn_points())
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicles.append(vehicle)

        # https://carla.readthedocs.io/en/latest/cameras_and_sensors
        # get the blueprint for this sensor
        blueprint = blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        blueprint.set_attribute('image_size_x', f'{IMG_WIDTH}')
        blueprint.set_attribute('image_size_y', f'{IMG_HEIGHT}')
        blueprint.set_attribute('fov', '110')
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
        sensors.append(sensor)
        sensor.listen(lambda data: process_img(data))

        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        time.sleep(5)

    finally:
        for sensor in sensors:
            sensor.destroy()
        for vehicle in vehicles:
            vehicle.destroy()
    print('Done running.')

main()