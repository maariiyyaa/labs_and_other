#!/usr/bin/env python
# coding: utf-8

from PIL import Image
from matplotlib import pyplot as plt
from em_functions import *
from json import load
import time

def main():

    t = time.process_time()

    with open('conf.json') as config:
        conf_data = load(config)
        path_to_file = conf_data['path_to_file']
        epsilon = conf_data['epsilon']
        size = (conf_data['height'], conf_data['width'])
        EM_iters = conf_data['EM_iters']
        sampling_iters = conf_data['sampling_iters']

    # load the image and convert to numpy array
    image = Image.open(path_to_file).resize(size)
    plt.imshow(image)
    plt.show()
    img_tenzor = np.asarray(image)
    # sumarize shape
    print(f' img shape {img_tenzor.shape}')

    params_1, params_2  = EM_fit(img_tenzor, EM_iters)
    print(f'class 1 params:{params_1},\nclass 2 params:{params_2}')
    marked_image_data = EM_predict(img_tenzor, params_1, params_2, sampling_iters, epsilon)

    plt.imshow(marked_image_data)
    plt.show()

    print(f'time:  {time.process_time() - t}')

if __name__ == '__main__':
    main()
