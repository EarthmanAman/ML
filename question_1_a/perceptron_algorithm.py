"""
A script to implement the perceptron in machine learning
"""

import numpy as np


class Perceptron:
    """
        
        Parameters
        ------------
            learning rate = number
            epochs = maximum number of iteration

        Methods
        ---------
            fit(self, X, y) = use to fit / train the model
            predict(self, X, y) = It is used to predict
            _unit_step_func(self, x) = Its a helper function for the method predict(self, X)

    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.n_iters = epochs
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    
    def fit(self, Input, target):
         """
            Parameters
            -----------
                Input - Input values
                target - target output
            Return
            --------
                No return

        """

        n_samples, n_features = Input.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in target])

        for _ in range(self.n_iters):
            
            for index, x_i in enumerate(Input):

                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                
                # Perceptron update rule
                update = self.lr * (y_[index] - y_predicted)

                self.weights += update * x_i
                self.bias += update

    def predict(self, Input):
         """
            Parameters
            -----------
                Input - Input values

            Return
            --------
                predicted value

        """
        linear_output = np.dot(Input, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)
        

