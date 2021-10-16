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


    # initialize step for epsilon devision
    step = 3
    # initialize height x width x 2 matrix
    marked_image_data  = zeros(shape=(*(shape)[:2], 2) , dtype=float)

    for i in range(shape[0]):
        for j in range(shape[1]):
            marked_image_data[i][j] = log_likelyhood(img_tenzor[i][j], segment_1_mean, segment_2_mean, segment_1_cov, segment_2_cov)

    marked_image_data = define_arcs_structure(marked_image_data, 0.2, shape)       
    marked_image_data = diffusion(marked_image_data, shape, 2)
    eps = find_max_eps(marked_image_data)
    print(f'max eps= {eps}')

    while eps-eps/step>0.1:
        markup_exists, matrix_copy1 = cut_superfluous_arcs(eps/step, marked_image_data)
        try:
            if markup_exists == 0:
                matrix_copy1 = matrix_copy2   
                final_marks = mark_final_image(matrix_copy1)    
                plt.imshow(final_marks)
                plt.show()
                break
        except:
            print('step should be less')
            step = step*0.8
            continue
        else:
            matrix_copy2 = copy.deepcopy(matrix_copy1)

        eps = eps/step
        final_marks = mark_final_image(matrix_copy1)    
        plt.imshow(final_marks)
        plt.show()


    print(f'time:  {time.process_time() - t}')

if __name__ == '__main__':
    main()
