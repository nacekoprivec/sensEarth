import argparse
import json
import sys
import requests
import threading
import time
import logging
from datetime import datetime
import itertools  
from typing import Any, Dict, List
from multiprocessing import Process
from src.Test import Test
from src.consumer import ConsumerAbstract, ConsumerFile, ConsumerKafka

from src.AnomalyDetectorWrapper import AnomalyDetectorWrapper
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

from itertools import product

def ping_watchdog(process: Process) -> None:
    """
    Function to ping the watchdog at regular intervals.

    Args:
        process (Process): The child process to monitor.

    Returns:
        None
    """
    interval = 30 # ping interval in seconds
    url = "localhost"
    port = 5001
    path = "/pingCheckIn/Data adapter"

    while(process.is_alive()):
        print("{}: Pinging.".format(datetime.now()))
        try:
            r = requests.get("http://{}:{}{}".format(url, port, path))
        except requests.exceptions.RequestException as e:
            logging.warning(e)
        else:
            logging.info('Successful ping at ' + time.ctime())
        time.sleep(interval)

def custom_scorer(estimator, X, y=None):
    return estimator.score(X, y)

def start_consumer(args: argparse.Namespace) -> None:
    """
    Function to start the consumer based on the command line arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """
    if(args.data_file):
        consumer = ConsumerFile(configuration_location=args.config)

    elif args.test:
        test_instance = Test(configuration_location=args.config)
        test_instance.read_streaming_data(args.data)
        return test_instance

    else:
        consumer = ConsumerKafka(configuration_location=args.config)
        
    print("=== Service starting ===", flush=True)
    consumer.read()

def main() -> None:
    """
    Main entry point of the program.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    parser = argparse.ArgumentParser(description="consumer")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config1.json",
        help=u"Config file located in ./config/ directory."
    )

    parser.add_argument(
        "-f",
        "--file",
        dest="data_file",
        action="store_true",
        help=u"Read data from a specified file on specified location."
    )

    parser.add_argument(
        "-fk",
        "--filekafka",
        dest="data_both",
        action="store_true",
        help=u"Read data from a specified file on specified location and then from kafka stream."
    )

    parser.add_argument(
        "-w",
        "--watchdog",
        dest="watchdog",
        action='store_true',
        help=u"Ping watchdog",
    )

    parser.add_argument(
        "-t", 
        "--test", 
        dest="test", 
        default="config1.json",
        help=u"Config file located in ./config/ directory."
    )
    # Display help if no arguments are defined
    if (len(sys.argv) == 1):
        parser.print_help()
        sys.exit(1)

    # Parse input arguments
    args = parser.parse_args()

    # Ping watchdog every 30 seconds if specified
    if (args.watchdog):
        process = Process(target=start_consumer, args=(args,))
        process.start()
        print("=== Watchdog started ==", flush=True)
        ping_watchdog(process)
    else:
        start_consumer(args)


if (__name__ == '__main__'):
    main()
