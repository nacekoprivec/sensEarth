from abc import ABC, abstractmethod
import csv
import json
import sys

from typing import Any, Dict, List
sys.path.insert(0,'./src')

# Algorithm imports
from algorithms.anomaly_detection import AnomalyDetectionAbstract
from algorithms.border_check import BorderCheck
from algorithms.welford import Welford
from algorithms.ema import EMA
from algorithms.ema_percentile import EMA_Percentile
from algorithms.filtering import Filtering
from algorithms.isolation_forest import IsolationForest
from algorithms.gan import GAN
from algorithms.pca import PCA
from algorithms.hampel import Hampel
from algorithms.linear_fit import LinearFit
from algorithms.combination import Combination
from algorithms.trend_classification import Trend_Classification
from algorithms.cumulative import Cumulative
from algorithms.macd import MACD
from algorithms.clustering import Clustering
from algorithms.percentile import Percentile

from algorithms.rrcf_trees import RRCF_trees

# TODO: imports
# from algorithms.fb_prophet import fb_Prophet


from json import loads
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
import pandas as pd
import datetime
from consumer import ConsumerAbstract

import csv
import json
import sys
import datetime
import pandas as pd
from typing import Any, Dict, List
from consumer import ConsumerAbstract

class Test(ConsumerAbstract):
    def __init__(self, conf: Dict[Any, Any] = None, configuration_location: str = None) -> None:
        super().__init__(configuration_location=configuration_location) # for clustering testing configuration_location="clustering.json" | isol_forest "isolation_forest.json"
        self.data_buffer = []  # Store manually inserted data

        self.conf = conf
        self.anomaly_counter = [] 
        self.indx = 0
        self.y_true = []
        self.X = []
        self.y = []
        
        self.pred_is_anomaly = 0
        
        if conf:
            self.configure(con=conf)
        elif configuration_location:
            with open("configuration/" + configuration_location) as data_file:
                conf = json.load(data_file)
            self.configure(con=conf)
        else:
            print("No configuration was given")

    def configure(self, con: Dict[Any, Any]) -> None:
        self.file_name = con.get("file_name", None)
        self.file_path = self.file_name

        self.anomaly_names = con["anomaly_detection_alg"]
        self.anomaly_configurations = con["anomaly_detection_conf"]

        self.filtering = con.get("filtering", None)

        assert len(self.anomaly_names) == len(self.anomaly_configurations),\
            "Number of algorithms and configurations do not match"

        self.anomalies = []
        for i, anomaly_name in enumerate(self.anomaly_names):
            anomaly = eval(anomaly_name)
            anomaly.configure(self.anomaly_configurations[i],
                              configuration_location=self.configuration_location,
                              algorithm_indx=i)
            self.anomalies.append(anomaly)

    def read(self) -> None:
        if(self.file_name[-4:] == "json"):
            self.read_JSON()
        elif(self.file_name[-3:] == "csv"):
            self.read_csv()
        else:
            print("Consumer file type not supported.")
            sys.exit(1)

    def read_JSON(self):
        with open(self.file_path) as json_file:
            data = json.load(json_file)
            tab = data["data"]
        for d in tab:
            for i, a in enumerate(self.anomalies):
                self.anomalies[i].message_insert(d)

    def read_csv(self):
        with open(self.file_path, 'r') as read_obj:
            csv_reader = csv.reader(read_obj)

            header = next(csv_reader)

            try:
                timestamp_index = header.index("timestamp")
            except ValueError:
                timestamp_index = None
            other_indicies = [i for i, x in enumerate(header) if ((x != "timestamp") and (x != "label") and (x != "labelInfo"))]

            # Iterate over each row in the csv using reader object
            for row in csv_reader:
                d = {}
                if(timestamp_index is not None):
                    timestamp = row[timestamp_index]
                    try:
                        timestamp = float(timestamp)
                    except ValueError:
                        pass
                    d["timestamp"] = timestamp

                try:
                    ftr_vector = [float(row[i]) for i in other_indicies]
                except:
                    ftr_vector = [row[i] for i in other_indicies]

                d["ftr_vector"] = ftr_vector
                message = d

                for i, a in enumerate(self.anomalies):
                    if(self.filtering is not None and eval(self.filtering[i]) is not None):
                        #extract target time and tolerance
                        target_time, tolerance = eval(self.filtering[i])
                        message = self.filter_by_time(d, target_time, tolerance)

                    if message is not None:    
                        self.data_buffer.append((row, self.anomalies[i].message_insert(d)))
                        #self.calculate_confusion_matrix()

    def read_streaming_data(self, d):
        message = d
        for i, a in enumerate(self.anomalies):
            if(self.filtering is not None and eval(self.filtering[i]) is not None):
                #extract target time and tolerance
                target_time, tolerance = eval(self.filtering[i])
                message = self.filter_by_time(d, target_time, tolerance)

            if message is not None:    
                self.data_buffer.append((message, self.anomalies[i].message_insert(d)))
                #self.calculate_confusion_matrix()
                self.classify_data()
                #print("printing data buffer", self.data_buffer[0], "\n", self.pred_is_anomaly, "\n")

    def classify_data(self) -> None:
        """Classifies the latest anomaly detection result."""

        data = self.data_buffer[-1]

        if data[0] is None or data[1] is None:
            print("Prediction data is missing or malformed.", data)
            return 
        
        predicted_anomaly = data[1][0].split(":")[0]
        
        if predicted_anomaly == "Error":
            self.pred_is_anomaly = 1
    
    def filter_by_time(self, message, target_time, tolerance):
        #convert to timedelta objects

        # Convert unix timestamp to datetime format (with seconds unit if
        # possible alse miliseconds)

        #print('filering; timestamp: ' + str(message['timestamp']), flush=True)
        try:
            timestamp = pd.to_datetime(message['timestamp'], unit="s")
        except(pd._libs.tslibs.np_datetime.OutOfBoundsDatetime):
            timestamp = pd.to_datetime(message['timestamp'], unit="ms")

        # timestamp = pd.to_datetime(message.value['timestamp'], unit='s')
        time = timestamp.time()
        target_time = datetime.time(target_time[0], target_time[1], target_time[2])
        tol = datetime.timedelta(hours = tolerance[0], minutes = tolerance[1], seconds = tolerance[2])
        date = datetime.date(1, 1, 1)
        datetime1 = datetime.datetime.combine(date, time)
        datetime2 = datetime.datetime.combine(date, target_time)

        # Return message only if timestamp is within tolerance
        # print((max(datetime2, datetime1) - min(datetime2, datetime1)))
        # print(tol)
        #print('razlika: ' + str((max(datetime2, datetime1) - min(datetime2, datetime1))), flush=True)
        if((max(datetime2, datetime1) - min(datetime2, datetime1)) < tol):
            #print('filtriral!', flush=True)
            return(message)
        else:
            #print('Nisem :(', flush=True)
            return(None)


