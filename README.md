# KU Data Preprocessing Package

## 1. Repository Structure
```sh
.
├── dataset
│   ├── ecg_mitbih_test.csv
│   ├── imputed_data
│   │   └── ecg_mitbih_test_imputed.csv
│   ├── decomposed_data
│   │   ├── ecg_mitbih_test_imputed.csv
│   │   └── trend_decomposed.csv
│   └── synchronized_data
│       └── synchronized_dtw.csv
├── imputation.py
├── seasonal_trend_decomposition.py
├── synchronization.py
└── README.md
```

## 2. Preprocessing module
### 2.1 Missing Value (NA) Imputation
#### 2.1.1 Supported Options & Sample Usage
Impute the missing values in a dataset and save the result.
- Simple Imputation with `mean, median, most_frequent, constant` value [[description]](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
```Python
# Sample Usage
python imputation.py --data_path='./dataset/ecg_mitbih_test.csv' \
                     --option='simple' \
                     --strategy='mean' \
                     --output_path='./dataset/imputed_data/ecg_mitbih_test_imputed.csv'
```

- KNN Imputation [[description]](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
```Python
# Sample Usage
python seasonal_trend_decomposition.py --data_path='./dataset/ecg_mitbih_test.csv' \
                                       --option='knn' \
                                       --n_neighbors=5 \
                                       --output_path='./dataset/imputed_data/ecg_mitbih_test_imputed.csv'
```
- MICE Imputation [[description]](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn-impute-iterativeimputer)
```Python
# Sample Usage
python imputation.py --data_path='./dataset/ecg_mitbih_test.csv' \
                     --option='mice' \
                     --strategy='mean' \
                     --output_path='./dataset/imputed_data/ecg_mitbih_test_imputed.csv'
```
#### 2.1.2 Testing imputation module by adding random NAs to temporary dataset.
Just add `--test_module` argument to the command-line for testing the module.
If ``--test_module` argument is given, `imputation.py` automatically adds random NAs to the dataset and then continues to impute the missing values.
```Python
* Sample Usage
python imputation.py --data_path='./dataset/ecg_mitbih_test.csv' \
                     --option='simple' \
                     --strategy='mean' \
                     --output_path='./dataset/imputed_data/ecg_mitbih_test_imputed.csv'
                     --test_module
```

### 2.2 Seasonal Trend Decomposition and Prediction (STL)
#### 2.2.1 Seasonal Trend Detection using Seasonal-Trend LOESS (STL)
- STL Decomposition [[description]](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.DecomposeResult.html#statsmodels.tsa.seasonal.DecomposeResult)

#### 2.2.1 Diagnosis of Patterns in Time-Series data
- Auto Arima [[description]](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)
```Python

# Sample Usage
python imputation.py --data_path='./dataset/machine_temperature_system_failure.csv' \
                     --seasonal_output_path='./dataset/decomposed_data/seasonal_decomposed.csv'
                     --trend_output_path='./dataset/decomposed_data/trend_decomposed.csv'
```

### 2.3 Synchronization using DTW and soft-DTW
- DTW [[description]](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html)
- soft-DTW [[description]](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.soft_dtw.html)
```Python

# Sample Usage
python synchronization.py --data_path='./dataset/power_voltage.csv' \
                          --dtw_output_path='./dataset/synchronized_data/synchronized_dtw.csv'\
                          --plot_output_path='./dataset/synchronized_data'\
                          --option='dtw'\
                          --distance=2


