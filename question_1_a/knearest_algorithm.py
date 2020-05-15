import numpy as np
from collections import Counter


def euclidean_distance(value1, value2):
        return np.sqrt(np.sum((value1 - value2)**2))

"""

A Script which implement the k-nearest algorithm

"""

class KNearestNeigh:
    """
        
        Parameters
        ------------
            The class takes the k value as parameter

        Methods
        ---------
            fit(self, X, y) = use to fit / train the model
            predict(self, X, y) = It is used to predict
            _predict(self, x) = Its a helper function for the method predict(self, X)

    """
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """
            Parameters
            -----------
                X - Input values
                y - target output

            Return
            --------
                No return

        """
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        """
            Parameters
            -----------
                X - Input values

            Return
            --------
                A numpy ndarray of the predicted values

        """

        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):

        """
            Parameters
            -----------
                x - one input value
            Return
            --------
                most common label

        """

        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]