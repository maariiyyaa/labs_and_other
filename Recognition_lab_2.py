import json
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


distribution = lambda x: ((1 - np.sign(x))/2).astype(np.int)
class Perceptron():

    def __init__(self):
        return
    
    def make_expanded_vector(self, vector, dim):
        """ Function define support vector for correction no positive defined matrix
            :params vector, dim: sample vector,  vector dimention
            :return: expanded vector of sample vector

        >>> a = Perceptron()
        >>> a.make_expanded_vector([2  , 3], 2)
        array([4, 6, 6, 9, 2, 3, 1])
       
        >>> a.make_expanded_vector([2, 3, 1], 3)
        array([4, 6, 2, 6, 9, 3, 2, 3, 1, 2, 3, 1, 1])
        """
        ksi = np.einsum('i,j->ij', vector, vector).ravel()
        for vector_index in range(dim):    
            ksi = np.append(ksi, vector[vector_index])
        ksi = np.append(ksi, 1)
        return ksi
    
    def make_support_vector(self, list_, dim):
        """ Function define support vector for correction no positive defined matrix
            :params list_, dim: vector, vector dimention
            :return: support vector

        >>> a = Perceptron()
        >>> a.make_support_vector([2, 3, 4], 3)
        array([ 4,  6,  8,  6,  9, 12,  8, 12, 16,  0,  0,  0,  0])
       
        >>> a.make_support_vector([5] , 1)
        array([25,  0,  0])
        """
        support_vector = np.einsum('i,j->ij',list_,list_).ravel()
        for j in range(dim + 1):    
            support_vector = np.append(support_vector, 0)
        return support_vector
        
    def check_positive_defined_matrix(self, dim):
        """ Function define and correct no positive defined matrix
            :params dim: train set dimention 
            :return: none
        """
        A = self.alpha[:dim**2]
        matrix = A.reshape(dim, dim)
        eigen_values, eigen_vectors = np.linalg.eigh(matrix)

        #findind non-positive eigenvalues of matrix
        if (list(filter(lambda x: np.any(x <= 0), eigen_values))):
            negative_eigenval_index = list(eigen_values).index(list(filter(lambda x: np.any(x <= 0), eigen_values)))
            etta = self.make_support_vector(eigen_vectors[:,negative_eigenval_index], dim)
            self.alpha += etta
    


    def fit(self, X, target):
        """ Function fits the model
            :params X, target: sample values, target values for X
            :return: fited model
        """
        dim = X.shape[1]
        self.alpha = np.zeros(dim**2+dim+1)
        self.steps = 0
        correction_isneeded = True
        while (correction_isneeded):
            correction_isneeded = False
            for row_index in range(len(X)):
                ksi = self.make_expanded_vector(X[row_index], dim)
                if (np.dot(ksi,self.alpha) >= 0 and target[row_index] == 1):
                    correction_isneeded = True
                    self.alpha -= ksi
                    self.steps += 1
                
                if (np.dot(ksi,self.alpha) <= 0 and target[row_index] == 0):
                    correction_isneeded = True
                    self.alpha += ksi
                    self.steps += 1 
                
            self.check_positive_defined_matrix(dim)
        return self

    def predict(self, X):
        """ Function evaluate
            :params X: sample vectors
            :return: evaluated values for X

        """
        ksi = [0] * len(X)
        dim = X.shape[1]
        for k in range(X.shape[0]):
            ksi[k] = self.make_expanded_vector(X[k], dim)
        ksi = np.array(ksi)
        return distribution(ksi.dot(self.alpha))

def main():
    with open('train_02.json') as config:
        conf_data = json.load(config)
        frame_inside = DataFrame.from_dict(conf_data['inside'])
        frame_outside = DataFrame.from_dict(conf_data['outside'])
    frame_inside['target'] = 1
    frame_outside['target'] = 0
    frame = frame_inside.append(frame_outside, ignore_index=True)
    #frame.rename(columns={0: 'x1', 1: 'x2'}, inplace=True)

    sample = frame.iloc[:, 0:-1].to_numpy()
    targets = frame.iloc[:, -1].to_numpy()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(sample, targets, test_size=0.3, random_state=1)

    clasifier = Perceptron()
    clasifier.fit(Xtrain, Ytrain)
    prediction = clasifier.predict(Xtest)
    first_group_indexes = np.where(prediction == 1)
    print ('true_values: {},\n prediction: {},\n accuracy: {},\n first_group_indexes: {}, \n alpha: {}'\
           .format(Ytest, prediction, accuracy_score(Ytest, prediction),first_group_indexes, clasifier.alpha))

    # plot_decision_regions(Xtest, Ytest, clf=clasifier, legend=2)
    # # Adding axes annotations
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.title('Perceptron on sample of Gaussian distribution')
    # plt.show()


if __name__ == '__main__':
    main()
    import doctest
    doctest.testmod()



