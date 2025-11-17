from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import numpy as np
from src.Test import Test

class AnomalyDetectorWrapper(BaseEstimator):
    def __init__(self, anomaly_class=None, fixed_params=None, **kwargs):
        self.anomaly_class = anomaly_class
        self.fixed_params = fixed_params or {}
        self.kwargs = kwargs  # Flattened config params
        self.file = "data/ads-1.csv"
        self.f1 = 0.0
        self.y_true_ = None

    def fit(self, X, y=None):
        # Merge fixed and variable params
        combined_conf = {**self.fixed_params, **self.kwargs}

        #for gan
        #keys_to_move = ["model_name", "N_shifts", "N_latent", "K", "len_window"]

        #for isolation forest
        #keys_to_move = ["max_samples", "max_features", "model_name"]

        #for pca
        #keys_to_move = ["max_samples", "max_features", "model_name", "N_components"]


        #for training
        #train_conf = {k: combined_conf.pop(k) for k in keys_to_move if k in combined_conf}

        config = {
            "file_name": self.file,
            "anomaly_detection_alg": [self.anomaly_class],
            "anomaly_detection_conf": [{
            **combined_conf,

            #for training 
            #"train_conf": train_conf,
        }],
        }
        
        test_instance = Test(config)
        test_instance.read()
        test_instance.confusion_matrix()
        self.f1 = test_instance.F1
        self.y_true_ = np.array(test_instance.y_true)
        return self

    def predict(self, X):
        return []

    def score(self, X, y=None):
        return self.f1

    def get_params(self, deep=True):
        return {
            'anomaly_class': self.anomaly_class,
            'fixed_params': self.fixed_params,
            **self.kwargs  # flattened dynamic parameters
        }

    def set_params(self, **params):
        self.anomaly_class = params.pop('anomaly_class', self.anomaly_class)
        self.fixed_params = params.pop('fixed_params', self.fixed_params)
        self.kwargs = params  # all remaining are dynamic hyperparams
        return self
