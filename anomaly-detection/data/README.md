# Data folder

Keep pilot specific data into `pilots/` folder. This data should not be added to the repository.

For experiments and testing use experimental artificial labelled datasets in corresponding files in the main directory.

## CSV files

CSV files should contain the following field:

* `timestamp` - defining timestamp
* `label` - `True` or `False` defining if a field is an anomaly or not - field is optional for evaluation datasets
* `labelInfo` - custom field defining extend of the anomaly for labelled dataset

Any other fields will be interpreted as columns of a feature vector.

## Test datasets

### Artificial data-set for testing time-series additive outlier detecion methods.

+ **Date**: June 2017
+ **DOI:** 10.13140/RG.2.2.23165.97760
+ **License:** LicenseCC BY-SA 4.0
+ **Link:** [ResearchGate](https://www.researchgate.net/publication/317721142_Artificial_data-set_for_testing_time-series_additive_outlier_detecion_methods)
+ **Files:** ads{i}.csv, *i = {1, ..., 9}*

**Related paper**: Kenda, Klemen, and Dunja Mladenić. "Autonomous sensor data cleaning in stream mining setting." _Business Systems Research: International journal of the Society for Advancing Innovation and Research in Economy 9.2_ (2018): 69-79.
