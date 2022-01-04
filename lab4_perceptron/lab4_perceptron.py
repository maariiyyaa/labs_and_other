#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.animation as animation
from pylab import *
from sklearn.utils import shuffle


# Dataset generation
class DatasetCreator:
    
    def __init__(self, mean, cov, d_coef=0.1, eps_coef=0.1):
        """
        Constructs class instance
        
        :param mean: mean of normal distribution
        :param cov: cov matrix of normal distribution
        :param d_coef: distribution diameter
        :param eps_coef: boundary
        """
        
        self.mean = mean
        self.cov = cov
        self.d_coef = d_coef
        self.eps_coef = eps_coef
        
        
    @staticmethod
    def _division_formula(x, mean, cov):
        """
        Apply formula to separate points into 2 classes
        
        :param x: a point of normal distribution
        :param mean: mean of normal distribution
        :param cov: cov matrix of normal distribution
        
        return: formula result
        """
        
        res = ((x - mean).T).dot(np.linalg.inv(cov)).dot((x - mean).T)
        return res
    
    @staticmethod
    def _get_class_idxs(idx, needed_points_amount):
        """
        Select random indexes from
        
        :param idx: whole list of indexes
        :param needed_points_amount: amount of indexes to be selected
        
        return: array of selected indexes
        """
        
        random_indices = np.random.choice(idx, size=needed_points_amount, replace=True)
        return random_indices

    def generate_sample(self, points_class1 = 1000, points_class2 = 1000,):
        """
        Generates samples of normal distribution for 2 classes
        
        :param points_class1: points amount in class 1
        :param points_class2: points amount in class 2
        
        return: ndarrays of class1 and class2 points
        
        """
        
        #generate random normal points
        normal_points = np.random.multivariate_normal(self.mean, self.cov, 10*points_class1)
        result_array = []
        for i in normal_points:
            result_array.append(self._division_formula(i, self.mean, self.cov))
        #select random 1000 points of class1
        class1_idx = np.concatenate(np.argwhere(result_array >= ((sum(self.cov.diagonal()) * self.d_coef)*(1+self.eps_coef))))
        class1_idx = self._get_class_idxs(class1_idx , points_class1)
        X1 = normal_points[class1_idx, :]  
        #select random 1000 points of class2
        class2_idx = np.concatenate(np.argwhere(result_array <= (sum(self.cov.diagonal()) * self.d_coef*(1-self.eps_coef))))
        class2_idx = self._get_class_idxs(class2_idx , points_class1)
        X2 = normal_points[class2_idx, :]

        return X1, X2

    @staticmethod
    def create_dataframe(sample1, sample2): 
        """
        Create a dataframe from ndarrays of class1 and class2 points
        
        :param sample1: ndarray of class1 points
        :param sample2: ndarray of class2 points
        
        return: pandas dataframe
        """
        
        columns_names = ['x{}'.format(i) for i in range(sample1.shape[1])]
        df1 = pd.DataFrame(sample1, columns = [*columns_names])
        df1['target'] = 0
        df2 = pd.DataFrame(sample2, columns = [*columns_names])
        df2['target'] = 1
        return  df1.append(df2)

    @staticmethod
    def data_shuffle(data):
        """
        Shuffle a dataframe
        
        :param data: pandas dataframe
        
        return: shuffled pandas dataframe, shuffled X, shuffled y
        
        """
        
        data_shuffled = shuffle(data, random_state=42)
        X = data_shuffled.drop('target', 1)
        y = data_shuffled['target']
        return data_shuffled, X, y
    
    @staticmethod
    def plot_data(X1, X2, figsize=(20,20)):
        """
        Create a 3D plot of generated samples
        
        :param X1: ndarray of class1 points
        :param X2: ndarray of class2 points
        :param figsize: figure size
        """
        
        fig = plt.figure(figsize=figsize)
        ax= fig.add_subplot(111, projection='3d')
        ax.scatter3D(X1[:,0], X1[:,1], X1[:,2], color='yellow' );
        ax.scatter3D(X2[:,0], X2[:,1], X2[:,2], color='blue');
        plt.show()



#  Model definition
class Perceptron():

    def __init__(self):
        """
        Constructs class instance
        :param n_iter: number of the iterations
        """
    
        self.n_iter = 0
        self.alpha = 0

    @staticmethod
    def _vectors_transformation(vector):
        """
        Transforms vector (ksi) to appropriate form according to specific rule
        :param vector: vector to be transformed
        :return: transformed vector
        """
        
        vector_ = np.einsum('i,j->ij', vector, vector).ravel()
        vector_ = np.concatenate((vector_, vector, [1]))
        return vector_

    @staticmethod
    def _eig_vector_transformation(vector):
        """
        Transforms eigen vector (eta) to appropriate form according to specific rule
        :param vector: vector to be transformed
        :return: transformed vector
        """
        
        vector_ = np.einsum('i,j->ij', vector, vector).ravel()
        vector_ = np.concatenate((vector_, np.zeros(len(vector)+1)))
        return vector_

    def _check_eig(self, alpha):
        """
        Finds eigen vectors and their transformed versions. 
        :param alpha: transformed vector from initial properties
        :param dim: dimension of the X
        :return: array with transformed eigen vectors and eigen values
        """
        
        A = np.reshape(self.alpha[:self.dim**2], newshape=(self.dim, self.dim))
        v, l = np.linalg.eigh(A)
        l_all = [l[:, i] for i in range(len(l))]
        eta_l = [self._eig_vector_transformation(l_all[i]) for i in range(len(l))]
        return eta_l, v

    def fit(self, X, y):
        """
        Training process
        :param X: array with x values to be trained on
        :param y: array with target classes
        """
        
        self.dim = X.shape[1]
        self.alpha = np.zeros(self.dim**2+self.dim+1)
        correction_isneeded = True
        while (correction_isneeded):
            self.n_iter +=1
            correction_isneeded = False
            for xi, target in zip(X, y):
                ksi_i = self._vectors_transformation(xi)
                alpha_ksi = np.dot(ksi_i,self.alpha)
                if alpha_ksi >= 0 and target == 1:
                    correction_isneeded = True
                    self.alpha -= ksi_i
                elif alpha_ksi <= 0 and target == 0:
                    correction_isneeded = True
                    self.alpha += ksi_i
            if not correction_isneeded:            
                eta_l, eig_vals = self._check_eig(self.alpha)
                for i in range(len(eig_vals)):
                    if eig_vals[i] <= 0:
                        self.alpha = self.alpha + eta_l[i]
                        correction_isneeded = True
                        break

    def _alpha_multiply(self, X):
        """
        Finds product between alpha and X
        :param X: array with x values
        :return: array with values corresponded to product between alpha and X
        """
        
        return np.dot(X, self.alpha)

    def predict(self, X):
        """
        Function to predict target values
        :param X: array with x values to predict on
        :return: array with predicted target values
        """
        
        ksi_pred = np.zeros((X.shape[0], X.shape[1]**2+X.shape[1]+1))
        k = 0
        for xi in X:
            ksi_pred_i = self._vectors_transformation(xi)
            ksi_pred[k] = ksi_pred_i
            k += 1
        return np.where(self._alpha_multiply(ksi_pred) >= 0.0, 0, 1)

    def find_dist_params(self):
        """
        Function to calculate mean and cov. matrix from found alpha
        """
        Sigma_inv = np.reshape(self.alpha[:self.dim**2], newshape=(self.dim, self.dim))
        B = np.array([b/(-2) for b in self.alpha[self.dim**2:self.dim**2+self.dim]])
        mu = np.linalg.solve(Sigma_inv, B)
        return mu, np.linalg.inv(Sigma_inv)

    def animate(self, X, y, filename='demo.mp4'):
        """
        Animate predictions for every change of alpha (first 500 iterations)
        :param X: array with x values to be trained on
        :param y: array with target classes
        """
        
        if X.shape[1] != 3:
            raise Exception(f'Dimention of elements of X should be 3, got {X.shape[1]}!')
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('auto')
        
        graph1, = ax.plot(X[:,0], X[:,1], X[:,2], linestyle="", marker="o", color='yellow')
        graph2, = ax.plot(X[:,0], X[:,1], X[:,2], linestyle="", marker="o", color='yellow')
        
        self.dim = X.shape[1]
        self.alpha = np.zeros(self.dim**2+self.dim+1)
        self.n_iter = 0
        def update_graph(i):
            self.n_iter +=1
            X_t = X[:self.n_iter, :]
            y_t = y[:self.n_iter]
            for xi, target in zip(X_t, y_t):
                ksi_i = self._vectors_transformation(xi)
                alpha_ksi = np.dot(ksi_i,self.alpha)
                if alpha_ksi >= 0 and target == 1:
                    correction_isneeded = True
                    self.alpha -= ksi_i
                elif alpha_ksi <= 0 and target == 0:
                    correction_isneeded = True
                    self.alpha += ksi_i
            Y_pred = self.predict(X)
            if np.argwhere(Y_pred == 0).any(): 
                X1 = X[np.concatenate(np.argwhere(Y_pred == 0))]
            else:
                X1 = X
            if np.argwhere(Y_pred == 1).any():
                X2 = X[np.concatenate(np.argwhere(Y_pred == 1))]
            else:
                X2 = X
            graph1.set_data(X1[:,0], X1[:,1])
            graph1.set_3d_properties(X1[:,2])
            graph2.set_data(X2[:,0], X2[:,1])
            graph2.set_3d_properties(X2[:,2])
            graph2.set_color('blue')
           
            return graph1, graph2,

        ani = animation.FuncAnimation(fig, update_graph, cache_frame_data = False,
                                      frames = 500, repeat_delay = 50, interval=200)
        writer = animation.writers['ffmpeg'](fps=10.0)
        print('saving may take some time')
        ani.save(filename, writer=writer, dpi=100)



def main():
    #create dataset
    #declare params
    mean = np.array([1,-2,3])

    cov= np.array([[10,4,1],
           [4, 12, 3],
           [1, 3, 10]])

    coef_for_d= 0.01
    coef_for_eps = 0.4
    number_of_points=1000

    dc = DatasetCreator(mean, cov, coef_for_d, coef_for_eps)
    X1, X2 = dc.generate_sample(points_class1=number_of_points, points_class2=number_of_points)
    dc.plot_data(X1, X2)

    df = dc.create_dataframe(X1, X2)
    df_shuffle, X, y = dc.data_shuffle(df)
    X_new = X.values
    y_new = y.values


    #model train and inference
    clasifier = Perceptron()
    clasifier.fit(X_new, y_new)
    prediction = clasifier.predict(X_new)

    print(f'accuracy of the classifier:  {accuracy_score(prediction, y_new)}')
    mean_pred, cov_pred = clasifier.find_dist_params()
    print(f'\nmean predicted: {mean_pred}')
    print(f'\ncov predicted:\n {cov_pred}')
    print(f'\nclasifier steps:    {clasifier.n_iter}')


    #Create an animation of classification
    #save a record as "demo.mp4"
    print('Create an animation of classification')
    clasifier.animate(X_new, y_new, filename='demo.mp4')


if __name__ == '__main__':
    main()



