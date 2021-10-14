#!/usr/bin/env python
# coding: utf-8
from PIL import Image
from matplotlib import pyplot as plt
# from numpy import *
from functions import *
from json import load
import time


def main():
    t = time.process_time()

    with open('conf.json') as config:
        conf_data = load(config)
        path_to_file = conf_data['path_to_file']
        epsilon = conf_data['epsilon']
        size = (conf_data['height'], conf_data['width'])
        iterations = conf_data['diff_iterations']

    # load the image and convert to numpy array
    image = Image.open(path_to_file).resize(size)
    img_tenzor = asarray(image)
    shape = img_tenzor.shape
    print(f'Image shape {shape}\n')



    # cut examples 2 for image segments and finf distribution parameters
    segment_1, segment_2 = img_tenzor[:50, :50], img_tenzor[250:, 250:]
    segment_1 = concatenate((segment_1), axis=0)
    segment_1_mean = segment_1.mean(axis=0)
    segment_1_cov = cov(segment_1, rowvar=False)

    print(f'segment_1 MEAN  \t{segment_1_mean}\n')
    print(f'segment_1 COV MATRIX\n {segment_1_cov}\n')
    print(f'segment_1 COV MATRIX shape \t {segment_1_cov.shape}')

    segment_2 = concatenate((segment_2), axis=0)
    segment_2_mean = segment_2.mean(axis=0)
    segment_2_cov = cov(segment_2, rowvar=False)

    print(f'segment_2 MEAN  \t{segment_2_mean}\n')
    print(f'segment_2 COV MATRIX\n {segment_2_cov}\n')
    print(f'segment_2 COV MATRIX shape \t {segment_2_cov.shape}')

    # initiate height x width x 2 matrix
    marked_image_data = zeros(shape=(*(shape)[:2], 2), dtype=float)

    for i in range(shape[0]):
        for j in range(shape[1]):
            #find log probabilities of a pixel belonging to classes
            marked_image_data[i][j] = log_likelyhood(img_tenzor[i][j], segment_1_mean, segment_2_mean, segment_1_cov,
                                                     segment_2_cov)

    marked_image_data = define_arcs_structure(marked_image_data, epsilon, shape)
    marked_image_data = diffusion(marked_image_data, shape, iterations)
    eps = find_max_eps(marked_image_data)
    print(f'max eps= {eps}')

    matrix_copy1 = zeros(shape=((2,3)) , dtype=float)
    matrix_copy2 = zeros(shape=((2,3)) , dtype=float)


    while eps-eps/2>2:
        matrix_copy1 = copy.deepcopy(marked_image_data)
        eps = eps/3
        print(f'eps= {eps}')
        markup_exists = 1
        update_exisis = 1
        remove_low_arcs(matrix_copy1, eps)
        while markup_exists != 0 and update_exisis != 0:
            matrix_copy2 = copy.deepcopy(matrix_copy1)
            markup_exists, update_exisis = find_markup(matrix_copy1, shape)
            print(markup_exists, update_exisis)
    
        final_marks = mark_final_image(matrix_copy2)    
        plt.imshow(final_marks)
        plt.show()

    print(f'time:  {time.process_time() - t}')

if __name__ == '__main__':
    main()
