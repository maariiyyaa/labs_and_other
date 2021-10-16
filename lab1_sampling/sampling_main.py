#!/usr/bin/env python
# coding: utf-8

from PIL import Image
from numba import jit, float32
from numpy import asarray, concatenate, cov, zeros
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from functools import reduce
from json import load
import time


def likelyhood(pixel_color_list, dist_segm1_color, dist_segm2_color):
    """
        params: pixel_color_list - list of pixel RGB color layers
                dist_segm1_color - Probability density function object for segment 1
                dist_segm2_color - Probability density function object for segment 2
        return: class and probabilities of belonging to classes
    """

    pdf_1 = dist_segm1_color.pdf(pixel_color_list)
    pdf_2 = dist_segm2_color.pdf(pixel_color_list)
    if pdf_1 > pdf_2:
        return 1, pdf_1, pdf_2  # marker color of 1 class and it's probability
    else:
        return 0, pdf_1, pdf_2  # marker color of 2 class and it's probability


@jit(nopython=True)
def find_neighbors(matrix, i, j):
    """
        params: matrix - (n x m x 3) matrix of pixel classes along with their probabilities
                i - index on the x-axis
                j - index on the y-axis
        return: neighbors - list of cross neighbors of the pixel
    """

    neighbors = [matrix[i + step[0], j + step[1], 0] for step in
                 [(0, -1), (0, 1), (-1, 0), (1, 0)]
                 if ((0 <= i + step[0] < len(matrix)) and (0 <= j + step[1] < len(matrix[0])))]
    return neighbors


@jit(nopython=True)
def sampling(matrix, shape, eps):
    """
        params: matrix - (n x m x 3) matrix of pixel classes along with their probabilities
                shape - matrix shape
                eps - Probability that the neighbor has different class
        return: matrix - updated matrix of pixel classes along with their probabilities
   """

    # new_matrix = zeros(shape=shape, dtype=float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            neighbors = find_neighbors(matrix, i, j)
            k1 = matrix[i, j][1] * reduce(lambda a, b: a * b, [(1 - eps) if x == 1 else eps for x in neighbors])
            k2 = matrix[i, j][2] * reduce(lambda a, b: a * b, [(1 - eps) if x == 0 else eps for x in neighbors])
            prob_k1, prob_k2 = k1 / (k1 + k2), k2 / (k1 + k2)
            matrix[i, j, 0] = (1 if prob_k1 > prob_k2 else 0)
            matrix[i, j, 1:] = prob_k1, prob_k2
    return matrix


def main():
    t = time.process_time()

    # define variables
    with open('conf.json') as config:
        conf_data = load(config)
        path_to_file = conf_data['path_to_file']
        epsilon = conf_data['epsilon']
        size = (conf_data['height'], conf_data['width'])

    # load the image and convert to numpy array
    image = Image.open(path_to_file).resize(size)
    img_tenzor = asarray(image)
    shape = img_tenzor.shape
    print(f'Image shape {shape}\n')

    # get image segments
    segment_1, segment_2 = img_tenzor[:50, :50], img_tenzor[250:, 250:]

    # find mean vector and covariance matrix of each distribution
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

    # get Probability density function object for segment_1 and segment_2
    dist_segm1_color = multivariate_normal(mean=segment_1_mean, cov=segment_1_cov)
    dist_segm2_color = multivariate_normal(mean=segment_2_mean, cov=segment_2_cov)

    # mark out pixels with class and probabilities of belonging to classes
    marked_image_data = zeros(shape=shape, dtype=float)

    for i in range(shape[0]):
        for j in range(shape[1]):
            marked_image_data[i][j] = likelyhood(img_tenzor[i][j], dist_segm1_color, dist_segm2_color)

    # sampling
    for i in range(101):
        marked_image_data = sampling(marked_image_data, shape, epsilon)
        if i % 20 == 0:
            plt.imshow(marked_image_data[:, :, 0])
            plt.show()

    print(time.process_time() - t)


if __name__ == '__main__':
    main()
