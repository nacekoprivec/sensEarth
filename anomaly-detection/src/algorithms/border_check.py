# global imports
from typing import Any, Dict, List

# local imports
from algorithms.anomaly_detection import AnomalyDetectionAbstract

class BorderCheck(AnomalyDetectionAbstract):
    """
        This algorithm works with univariate time series and checks
        if the value is over or close to given upper and lower limits.
    """

    # method properties
    UL: float
    LL: float
    warning_stages: List[float]
    name: str = "border_check"
    filtering: None

    def __init__(self, conf: Dict[Any, Any] = None) -> None:
        """
        Initializes the class with the given configuration.

        Args:
            conf (Dict[Any, Any], optional): The configuration dictionary. Defaults to None.

        Returns:
            None
        """
        super().__init__()
        if(conf is not None):
            self.configure(conf)


    def configure(self, conf: Dict[Any, Any] = None,
                  configuration_location: str = None,
                  algorithm_indx: int = None) -> None:
        """
        Configure the algorithm with the given parameters.

        Args:
            conf (Dict[Any, Any], optional): A dictionary containing the configuration parameters. Defaults to None.
            configuration_location (str, optional): The location of the configuration file. Defaults to None.
            algorithm_indx (int, optional): The index of the algorithm. Defaults to None.

        Returns:
            None: This function does not return anything.

        Raises:
            None: This function does not raise any exceptions.

        Algorithm specific parameters:
            LL (float): The lower limit parameter.
            UL (float): The upper limit parameter.
            warning_stages (List[float]): A list of warning stages, sorted in ascending order.
        """
        super().configure(conf, configuration_location=configuration_location,
                          algorithm_indx=algorithm_indx)

        # algorithm specific parameters
        self.LL = conf["LL"]
        self.UL = conf["UL"]
        self.warning_stages = conf["warning_stages"]
        self.warning_stages.sort()



    def message_insert(self, message_value: Dict[Any, Any]) -> Any:
        """
            A function to insert a message into the system, with optional filtering and normalization.
            It checks the feature vector, extracts value and timestamp, constructs the feature vector,
            normalizes the value, checks limits, and sets status and status code with corresponding
            warnings. It also visualizes the normalization output.

            Args:
                message_value (Dict[Any, Any]): A dictionary containing the message value.
                                                Expects message in the form
                                                { timestamp: timestamp, ftr_vector: [val1, val2, ...] }

            Returns:
                Any: A tuple containing the status and status code.

        """

        # passing the message to AnomalyDetectionAbstract
        super().message_insert(message_value)

        # check if this topic needs filtering (only used for data sources, where
        # we should observe only specific periods of time)
        if (self.filtering is not None and eval(self.filtering) is not None):
            # extract target time and tolerance
            target_time, tolerance = eval(self.filtering)
            message_value = super().filter_by_time(message_value, target_time, tolerance)


        # check feature vector
        if (not self.check_ftr_vector(message_value=message_value)):
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE

            # remember status for unit tests
            # TODO: check if this is really needed
            self.status = status
            self.status_code = status_code
            return status, status_code

        # extract value and timestamp
        timestamp = message_value["timestamp"]
        feature_vector = super().feature_construction(value=message_value['ftr_vector'],
                                                      timestamp=message_value['timestamp'])

        if (feature_vector == False):
            # This happens if the memory does not contain enough samples to
            # create all additional features.
            status = self.UNDEFINED
            status_code = self.UNDEFIEND_CODE
        else:
            # only use the first value in the feature vector
            # TODO: should be configurable which value to use
            value = feature_vector[0]

            # value normalization
            value_normalized = 2*(value - (self.UL + self.LL)/2) / \
                (self.UL - self.LL)
            status = self.OK
            status_code = self.OK_CODE

            # check limits
            if (value_normalized > 1):
                status = "Error: measurement above upper limit"
                status_code = -1
            elif (value_normalized < -1):
                status = "Error: measurement below lower limit"
                status_code = self.ERROR_CODE
            else:
                for stage in range(len(self.warning_stages)):
                    if (value_normalized > self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": measurement close to upper limit."
                        status_code = self.WARNING_CODE
                    elif (value_normalized < -self.warning_stages[stage]):
                        status = "Warning" + str(stage) + \
                            ": measurement close to lower limit."
                        status_code = self.WARNING_CODE
                    else:
                        break

            # remember status for unit tests
            # TODO: check if this is really needed
            self.status = status
            self.status_code = status_code

            # render ouput (via logger of for visualization)
            self.normalization_output_visualization(status=status,
                                                    status_code=status_code,
                                                    value=message_value["ftr_vector"],
                                                    timestamp=timestamp)

        # return status and status code
        return status, status_code
