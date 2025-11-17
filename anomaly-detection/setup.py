from setuptools import setup, find_packages

setup(
    name='anomaly_detection',
    version='0.1',
    packages=find_packages(exclude=["api", "api.*"]),

)
