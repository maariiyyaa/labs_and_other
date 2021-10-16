#!/usr/bin/env python
# coding: utf-8

from numpy import *
from numba import jit, njit, float32
import math
from itertools import product
import copy




@jit(nopython=True)
def log_pdf(x, mu, cov):
    """

        :param x: numpy array of pixel RGB color layers numbers
        :param mu: distribution mean
        :param cov: distribution covariance matrix
        :return: log of probability density function for distribution with mu and cov
    """
    size = len(x)
    if size == len(mu) and (size, size) == cov.shape:
        det = linalg.det(cov)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = math.log(1.0 / (math.pow((2 * math.pi), float(size) / 2) * math.pow(det, 1.0 / 2)))
        x_mu = x - mu
        inv = linalg.inv(cov)
        result = (- 0.5 * ((x_mu).dot(inv).dot(x_mu.T)))
        return norm_const + result
    else:
        raise NameError("The dimensions of the input don't match")


def log_likelyhood(pixel_color_list, segment_1_mean, segment_2_mean, segment_1_cov, segment_2_cov):
    """

        :param pixel_color_list: list of pixel RGB color layers
        :param segment_1_mean: distribution mean for segment 1
        :param segment_2_mean: distribution mean for segment 2
        :param segment_1_cov: distribution cov matrix for segment 1
        :param segment_2_cov: distribution cov matrix for segment 2
        :return: log probabilities of a pixel belonging to classes
    """
    pdf_1 = log_pdf(pixel_color_list, segment_1_mean, segment_1_cov)
    pdf_2 = log_pdf(pixel_color_list, segment_2_mean, segment_2_cov)

    return pdf_1, pdf_2


@jit(nopython=True)
def find_neighbors_coord(i, j, dim1, dim2):
    """

    :param i: i position of a pixel (i,j)
    :param j: j position of a pixel (i,j)
    :param dim1: matrix height
    :param dim2: matrix width
    :return: list of tuples of neighbors coordinates for pixel (i,j)
    """

    neighbors = []
    for row_step in (-1, 0, 1):
        for col_step in (-1, 0, 1):
            if row_step * col_step == 0 and col_step != row_step:
                if i + row_step == -1 or j + col_step == -1:
                    continue
                if i + row_step >= dim1 or j + col_step >= dim2:
                    continue
                neighbors.append((i + row_step, j + col_step))

    return neighbors


def define_arcs_structure(matrix, eps, shape):
    """

    :param matrix: n x m x 2 matrix
    :param eps:  probability that the neighbor has different class
    :param shape: matrix's shape
    :return: n x m of neighbor structure for each pixel
    """
    new_matrix = empty(shape=shape[:2], dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # for each pixel define empty structure
            new_matrix[i, j] = {}
            neighbors = find_neighbors_coord(i, j, *shape[:2])
            for n_idx in neighbors:
                # on defined structure set neighbor's coord. as key and empty structure as value
                new_matrix[i, j][n_idx] = {}
                for k_current, k_neigh in product((0, 1), repeat=2):
                    # update empty value. set a tuple of marks as a key and arc's weight as value
                    new_matrix[i,j][n_idx][(k_current, k_neigh)] = (
                                    (log(eps) if k_current != k_neigh else log(1-eps))
                                    +(matrix[i,j,k_current])/(len(neighbors))
                                    +(matrix[n_idx[0], n_idx[1]][k_neigh])/(len(find_neighbors_coord( n_idx[0], n_idx[1], *shape[:2])))
                    )
                    
    return new_matrix


def find_max_eps(matrix):
    """

    :param matrix: n x m matrix of neighbor structure for each pixel
    :return: difference between max arc and min arc of structure
    """
    # initialize min/max arcs
    max_arc = -1000
    min_arc = 1000
    for i, j in ndindex(matrix.shape[:2]):
        for k in list(matrix[i, j].values()):
            # find max per all arcs
            max_arc = max(max_arc, (max(list(k.values()))))
            # find min per all arcs
            min_arc = min(min_arc, (min(list(k.values()))))
    return max_arc - min_arc


def remove_low_arcs(matrix, eps):
    """

    :param matrix: n x m matrix of neighbor structure for each pixel
    :param eps: permissible arcs deviation from the max one
    :return: n x m matrix of neighbor structure with removed arcs for each pixel
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # assign pixel dict to find fester
            pair_dict = matrix[i][j]
            for neighbor in pair_dict.keys():
                # get max arc for pair
                max_arc_per_neighbor = max(pair_dict[neighbor].values())
                pair_dict_copy = pair_dict[neighbor].copy()
                for key, arc in pair_dict_copy.items():
                    if arc < (max_arc_per_neighbor - eps):
                        # remove an arc with low weight
                        pair_dict[neighbor].pop(key, 0)
    return matrix


def mark_final_image(matrix):
    """

    :param matrix: n x m matrix of neighbor structure for each pixel
    :return: n x m matrix as the final class markup of pixels where matrix(i,j) = 1 or 0
    """
    final_marks = empty(shape=matrix.shape[:2], dtype=float)
    for i, j in ndindex(matrix.shape[:2]):
        for neighbor, classes in matrix[i][j].items():
            # set a class for carrent pixel
            final_marks[i, j] = list(classes.keys())[0][0]
            # set a class for neighbor pixel
            final_marks[neighbor] = list(classes.keys())[0][1]
    return final_marks


def diffusion(structure, shape, iterations):
    """

    :param structure: n x m matrix of neighbor structure for each pixel
    :param shape: shape of structure
    :param iterations: number of diffusion's iterations
    :return: matrix of updated neighbor structure for each pixel
    """
    for it in range(iterations):
        for i in range(shape[0]):
            for j in range(shape[1]):
                # find neighbors arcs and their values, each element is neighbor's arcs
                neighbors_arcs = [dict(list(structure[i][j][t].items())) for t in structure[i][j]]
                # class1 = list of neighbors' arcs and their values only if pix (i, j) has class 0
                class1 = [list(neighbors_arcs[l].items())[:2] for l in range(len(neighbors_arcs))]
                # class2 = list of neighbors' arcs and their values only if pix (i, j) has class 1
                class2 = [list(neighbors_arcs[l].items())[2:] for l in range(len(neighbors_arcs))]
                # find max values of arcs for every neighbor and for each class
                # then find mean max for every class
                max_g1 = [max([class1[m][k][1] for k in range(2)]) for m in range(len(class1))]
                mean_max_g1 = mean(max_g1)
                max_g2 = [max([class2[m][k][1] for k in range(2)]) for m in range(len(class2))]
                mean_max_g2 = mean(max_g2)
                # find updating values for arcs according to the formula: g - max_g + mean_g
                up_c1 = [[list(neighbors_arcs[t].items())[m][1] - max_g1[t] + mean_max_g1 for m in range(2)] for t in
                         range(len(neighbors_arcs))]
                up_c2 = [[list(neighbors_arcs[t].items())[m][1] - max_g2[t] + mean_max_g2 for m in range(2, 4)] for t in
                         range(len(neighbors_arcs))]
                update_arcs = concatenate([up_c1, up_c2], axis=1)
                count = 0
                c_n = find_neighbors_coord(i, j, shape[0], shape[1])
                # update arcs' values in our structure
                for p in range(len(neighbors_arcs)):
                    for n, m in product((0, 1), repeat=2):
                        structure[i][j][c_n[p]][(n, m)] = update_arcs[p][count]
                        structure[c_n[p][0], c_n[p][1]][(i,j)][(m, n)] = update_arcs[p][count]
                        count += 1
                    count = 0
    return structure


def find_markup(structure, shape):
    """

    :param structure: n x m matrix of neighbor structure for each pixel
    :param shape: shape of structure
    :return: 1 if marup exist for current structure, else 0
    """
    flag = 0
    for i, j in ndindex((shape[0], shape[1])):
        # find two lists: 1) neighbors' indices
        #                 2) arcs from pix (i, j) to neighbors pixels (arcs keys and their values)
        neighbors = []
        arcs = []
        c_n = find_neighbors_coord(i, j, shape[0], shape[1])
        for n in range(len(c_n)):
            neighbors.append(list(structure[i][j].items())[n][0])
            arcs.append(list(structure[i][j].items())[n][1])
        arcs_copy = copy.deepcopy(arcs)
        # take all arcs for each neighbor
        for a in range(len(arcs_copy)):
            false_count = 0
            # take a copy of current neighbor arcs to pix (i, j)
            structure_copy = copy.deepcopy(structure[neighbors[a]][(i, j)])
            # take each arc key in list of arcs for each neighbor
            for arc_classes in arcs_copy[a].keys():
                # reverse arc
                tmp = arc_classes[::-1]
                # check if we have this arc from neighbor to pix (i, j)
                if tmp in list(structure_copy.keys()):
                    # if "yes" we need to check that we have arc from this class also to all other neighbors
                    for n in c_n:
                        count = 0
                        # we don't need to check again current neighbor
                        if n != neighbors[a]:
                            # find all first indices of arc keys to neighbor n
                            first_arcs_idx = [list(structure[i][j][n].keys())[k][0] for k in
                                              range(len(structure[i][j][n].keys()))]
                            if arc_classes[0] not in first_arcs_idx:
                                # if there isn't arc from current class to another neighbor then we will delete this arc
                                # and if we don't have any arcs that are starting from one class to all neighbors then we
                                # will say that we don't have a murkup
                                count += 1
                                structure[i][j][neighbors[a]].pop(arc_classes, 0)
                                flag = 1
                                if count == len(arcs_copy[a]):
                                    print("confusion: markup doesn't exist")
                                    return 0, flag
                    continue
                else:
                    # if we don't have current arc from neighbor to pix (i, j) then we need to delete this arc
                    # print('pix', (i, j), 'neighbor', nei[a], 'arc', arc_classes)
                    false_count += 1
                    # len_tmp = len(arcs_copy[a])
                    structure[i][j][neighbors[a]].pop(arc_classes, 0)
                    flag = 1
                    # and if we delete all arcs then we don't have a markup
                    if false_count == len(arcs_copy[a]):
                        print("markup doesn't exist")
                        return 0, flag
    return 1, flag


def cut_superfluous_arcs(eps, marked_image_data):
    """
    
    :param structure: n x m matrix of neighbor structure for each pixel
    :param eps: parameter epsilon
    :return: tuple: 1 if marup exist for current structure, else 0 and new matrix of neighbor structure for each pixel
    """
    print(f'eps= {eps}')
    markup_exists = 1
    update_exisis = 1
    matrix_copy1 = copy.deepcopy(marked_image_data)
    remove_low_arcs(matrix_copy1, eps)
    while markup_exists != 0 and update_exisis!=0:
        markup_exists, update_exisis = find_markup(matrix_copy1, shape)
        print(markup_exists, update_exisis)
    return markup_exists, matrix_copy1


