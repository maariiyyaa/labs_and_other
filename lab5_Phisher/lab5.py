import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

class Preprocessing:
    """
    Class for preprocessing images and creating datasets
    """
    def __init__(self, data, mean=0, cov=20):
        """
        Constructs class instance
        :param data: DataFrame with data
        :param mean: mean of the gaussian distribution for noise
        :param cov: covariance of the gaussian distribution for noise
        """
        self.data = data
        self.mean = mean
        self.cov = cov
    
    @staticmethod
    def rotate(image):
        """
        Rotates image on 90 degrees. It is needed to get correct orientation of images from dataset
        :param image: image array
        :return: rotated image
        """
        img_width, img_height = int(np.sqrt(image.shape[0])), int(np.sqrt(image.shape[0]))
        image = image.reshape([img_width, img_height])
        image = np.fliplr(image)
        image = np.rot90(image)
        return image

    def select_images(self, img_idxs):
        """
        Selects etalons from data
        :param img_idxs: symbols idxs to select symbols from data
        :return: array with selected symbols
        """
        x = self.data.iloc[:, 1:]
        y = self.data.iloc[:, 0]
        x_new = np.asarray(x)
        x_new = np.apply_along_axis(self.rotate, 1, x_new)
        x_new = x_new.astype('float32')
        return x_new[img_idxs, :, :]

    @staticmethod
    def show_etalons(etalons):
        """
        Plots etalons
        :param etalons: array with etalons
        """
        if etalons.ndim == 2:
            plt.imshow(etalons)
            plt.show()
        else:
            for i, et in enumerate(etalons):
                plt.subplot(2, round(len(etalons) / 2), i + 1)
                plt.axis("off")
                plt.imshow(et, cmap=plt.get_cmap('gray'))
            plt.show()

    def add_noise(self, array):
        """
        Adds gaussian noise to images
        :param array: array to which noise will be added
        :return: array with noise
        """
        for i, val in enumerate(array):
            array[i] = (val + np.random.normal(self.mean, self.cov)) % 255
        return array

    def create_dataset(self, rows, cols, etalons, one_img_length, num_of_etalons):
        """
        Creates two arrays: one with randomly chosed idxs of etalons
                            and another array with all images' pixels corresponded to randomly cjosed idxs
        :param rows: number of rows in dataset
        :param cols: len of one row (number of images in one row)
        :param etalons: array with etalons
        :param one_img_length: len of flatten array of one image
        :param num_of_etalons: number of etalons
        :return: array with randomly selected idxs of etalons, array with all images' pixels corresponded to randomly selected idxs
        """
        rows = rows
        row_length = cols
        dataset_size = rows * row_length

        numbers_list = np.random.randint(0, num_of_etalons, dataset_size)
        pixels_list = []
        for i in tqdm(numbers_list):
            pixels_list = np.append(pixels_list, np.reshape(etalons[i], newshape=(one_img_length,)))

        pixels_list = self.add_noise(pixels_list)
        return numbers_list, pixels_list


class Phisher:
    def __init__(self, one_img_length, num_of_etalons, img_width, img_height, test_rows, test_row_length, verbose=False):
        """
        Constructs class instance
        :param one_img_length: len of flatten array of one image
        :param num_of_etalons: number of etalons
        :param img_width: image width
        :param img_height: image height
        :param test_rows: number of rows in test dataset
        :param test_row_length: len of one test row (number of images in one test row)
        :param verbose: if verbose==True => all images with results will be shown
        """
        self.one_img_length = one_img_length
        self.num_of_etalons = num_of_etalons
        self.img_width = img_width
        self.img_height = img_height
        self.test_rows = test_rows
        self.test_row_length = test_row_length
        self.verbose = verbose

    @staticmethod
    def get_slice(array, start_point, length):
        """
        Gets slice from array
        :param array: array to get slice from
        :param start_point: start index to slice
        :param length: number of pixels included to slice
        :return: array from start point to start_point+length
        """
        return array[start_point: start_point + length]

    def fit(self, X, y):
        """
        Training process
        :param X: array with train images
        :param y: array with labels
        :return: vector alpha
        """
        alpha = np.zeros(self.one_img_length * self.num_of_etalons)
        correction_is_needed = True
        while (correction_is_needed):
            correction_is_needed = False
            for idx, label in enumerate(y):
                for item in range(self.num_of_etalons):
                    if item != label:
                        alpha_slice_1 = self.get_slice(alpha, self.one_img_length * label, self.one_img_length)
                        alpha_slice_2 = self.get_slice(alpha, self.one_img_length * item, self.one_img_length)
                        x_slice = self.get_slice(X, self.one_img_length * idx, self.one_img_length)
                        if alpha_slice_1.dot(x_slice) <= alpha_slice_2.dot(x_slice):
                            alpha[self.one_img_length * label: self.one_img_length * (label + 1)] = (
                                    alpha_slice_1 + x_slice
                            )
                            alpha[self.one_img_length * item: self.one_img_length * (item + 1)] = (
                                    alpha_slice_2 - x_slice
                            )
                            correction_is_needed = True
        return alpha

    @staticmethod
    def normalize_alpha(alpha):
        """
        Normalizes alpha to [0, 255]
        :param alpha: array of alpha
        :return: normalized alpha
        """
        max_element = np.linalg.norm(alpha, np.inf)
        alpha = alpha / max_element * 255
        alpha = (alpha + 255) / 2
        return alpha

    def create_alpha_img(self, alpha, etalons):
        """
        Creates, saves and shows img with etalons and alpha
        :param alpha: array with alpha
        :param etalons: array with etalons
        :return: array with all etalons as one array
        """
        alpha_result = np.zeros((self.img_width, self.img_height * self.num_of_etalons))
        for i in range(self.num_of_etalons):
            result = np.reshape(self.get_slice(alpha, self.one_img_length * i, self.one_img_length), (self.img_width, self.img_height))
            alpha_result[:, self.img_height * i:self.img_height * (i + 1)] = result

        all_etalons_img = np.zeros((self.img_width, self.img_height * self.num_of_etalons))
        for i in range(self.num_of_etalons):
            all_etalons_img[0:self.img_width, i * self.img_height:self.img_height * (i + 1)] = etalons[i]

        result_img = np.zeros((alpha_result.shape[0] * 2, alpha_result.shape[1]))
        result_img[:alpha_result.shape[0], :] = all_etalons_img
        result_img[alpha_result.shape[0]:2 * alpha_result.shape[0], :] = alpha_result
        cv2.imwrite('alpha.png', result_img)

        if self.verbose:
            plt.imshow(result_img, cmap=plt.get_cmap('gray'))
            plt.show()

        return all_etalons_img

    def predict_one_image(self, alpha, pixels_list):
        """
        Predicts label for one image
        :param alpha: array with alpha values
        :param pixels_list: array with images' pixels corresponded to selected idxs
        :return: predicted label
        """
        result = self.num_of_etalons - 1
        for i in range(self.num_of_etalons - 1):
            first_slice = self.get_slice(alpha, self.one_img_length * i, self.one_img_length)
            second_slice = self.get_slice(alpha, self.one_img_length * result, self.one_img_length)
            if first_slice.dot(pixels_list) > second_slice.dot(pixels_list):
                result = i
        return result

    def predict(self, alpha, pixels_list):
        """
        Predicts idxs of symbols
        :param alpha: array with alpha values
        :param pixels_list: array with images' pixels corresponded to selected idxs
        :return: array with predicted idxs of symbols
        """
        num_pred = self.test_rows * self.test_row_length
        predictions = np.zeros(num_pred)
        for i in range(num_pred):
            test_slice = self.get_slice(pixels_list, self.one_img_length * i, self.one_img_length)
            predictions[i] = self.predict_one_image(alpha, test_slice)
        return predictions

    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Calculates accuracy score
        :param y_true: array with true labels
        :param y_pred: array with predicted labels
        :return: accuracy score in %, array with 0 and 1 values: if 1 => correct prediction, 0 => incorrect prediction
        """
        check_pred = np.where(y_true == y_pred, 1, 0)
        return np.sum(check_pred) / len(y_pred) * 100, check_pred
    
    @staticmethod
    def expand_dim(array):
        """
        Converts 1 channel array to 3 channels array
        :param array: array to be converted
        :return: array with 3 channels
        """
        array = np.expand_dims(array, axis=2)
        array = np.tile(array, (1, 1, 3))
        return array

    def plot_result(self, y, pixels_list, y_pred, all_etalons_img, one_res_img_rows=20):
        """
        Plots result. Incorect predictions are underlined
        :param y: array with correct labels
        :param pixels_list: array with images' pixels corresponded to chosed idxs
        :param y_pred: array with 0 and 1 values: if 1 => correct prediction, 0 => incorrect prediction
        :param all_etalons_img: array with all etalons image in one image
        :param one_res_img_rows: number of rows to be plotted on one image
        """
        index = 0
        current_index = 0
        slice = (self.img_width * 2 + self.num_of_etalons)
        underline = np.zeros((self.num_of_etalons, self.img_height, 3))
        underline[:, :, 0] = 255
        all_etalons_img = self.expand_dim(all_etalons_img)
        for k in range(int(self.test_rows / one_res_img_rows)):
            height = slice * one_res_img_rows
            img = np.full((height, self.test_row_length * self.img_height, 3), 255)
            for i in range(one_res_img_rows):
                for j in range(self.test_row_length):
                    symbols = pixels_list[index:index + self.one_img_length].reshape((self.img_width, self.img_height))
                    symbols = self.expand_dim(symbols)
                    index += self.test_row_length
                    img[slice * i:slice * i + self.img_width, self.img_height * j:self.img_height * (j + 1)] = symbols
                    img[slice * i + self.img_width:slice * i + 2 * self.img_width,
                    self.img_height * j:self.img_height * (j + 1)] = all_etalons_img[:self.img_width, self.img_height * y[current_index]:self.img_height * y[current_index] + self.img_height]
                    if y_pred[current_index] == 0:
                        img[slice * i + 2 * self.img_width:slice * i + slice, self.img_height * j:self.img_height * (j + 1)] = underline
                    current_index += 1

            if self.verbose:
                plt.imshow(img, cmap=plt.get_cmap('gray'))
                plt.axis("off")
                plt.show()
            
            img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)
            cv2.imwrite('result_{}'.format(str(k)) + '.png', img)


if __name__ == '__main__':
    print('---main---')

    data = pd.read_csv('emnist-letters-train.csv')

    preprocess = Preprocessing(data)

    etalons = preprocess.select_images((196, 197, 198, 199, 200, 201, 204, 205, 210, 211))
    preprocess.show_etalons(etalons)

    train_rows, train_row_length = 10, 100
    test_rows, test_row_length = 20, 100

    img_width, img_height = etalons.shape[1], etalons.shape[2]
    one_img_length = img_height * img_width
    num_of_etalons = etalons.shape[0]

    print('Num of etalons = {}, each of shape {}'.format(num_of_etalons, (img_width, img_height)))

    print('Creating train dataset')
    train_numbers_list, train_pixels_list = preprocess.create_dataset(train_rows, train_row_length, etalons, one_img_length,
                                                                      num_of_etalons)
    print('Creating test dataset')
    test_numbers_list, test_pixels_list = preprocess.create_dataset(test_rows, test_row_length, etalons, one_img_length,
                                                                    num_of_etalons)

    phisher = Phisher(one_img_length, num_of_etalons, img_width, img_height, test_rows, test_row_length)

    print('Training process')
    alpha = phisher.fit(train_pixels_list, train_numbers_list)
    alpha = phisher.normalize_alpha(alpha)

    all_etalons_img = phisher.create_alpha_img(alpha, etalons)

    print('Prediction process')
    y_pred = phisher.predict(alpha, test_pixels_list)

    print('Evaluating')
    accuracy, check_pred = phisher.evaluate(test_numbers_list, y_pred)
    print('Accuracy: {}%'.format(accuracy))

    phisher.plot_result(test_numbers_list, test_pixels_list, check_pred, all_etalons_img)
