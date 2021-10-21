"""Missing Value Imputation module for Numeric dataset. 

#####
# Sample Usage
#####
python imputation.py --data_path='./dataset/ecg_mitbih_test.csv' \
                     --option='simple' \
                     --strategy='mean' \
                     --output_path='./dataset/imputed_data/ecg_mitbih_test_imputed.csv'

python imputation.py --data_path='./dataset/ecg_mitbih_test.csv' \
                     --option='simple' \
                     --strategy='constant' \
                     --fill_value=0 \
                     --output_path='./dataset/imputed_data/ecg_mitbih_test_imputed.csv'
                     
python imputation.py --data_path='./dataset/ecg_mitbih_test.csv' \
                     --option='knn' \
                     --n_neighbors=5 \
                     --output_path='./dataset/imputed_data/ecg_mitbih_test_imputed.csv'
                
#####
# Testing Module by Adding Random Noise
#####
python imputation.py --data_path='./dataset/ecg_mitbih_test.csv' \
                     --option='simple' \
                     --strategy='mean' \
                     --output_path='./dataset/imputed_data/ecg_mitbih_test_imputed.csv'
                     --test_module
"""

import numpy as np
import pandas as pd
import argparse

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

parser = argparse.ArgumentParser(description='Temporal One-class Anomaly Detection')
parser.add_argument('--data_path', type=str, default='./dataset/ecg_mitbih_test.csv')
parser.add_argument('--output_path', type=str, default='./dataset/ecg_mitbih_test_imputed.csv')
parser.add_argument('--header', action="store_true", default=None)
parser.add_argument('--option',  type=str, default='simple', help='imputation method')
parser.add_argument('--strategy', type=str, default=None, 
                    help='strategy for simple imputation. [mean, median, most_frequent, constant]')
parser.add_argument('--fill_value', type=float, default=None, 
                    help='If the strategy is “constant”, then replace missing values with fill_value.')
parser.add_argument('--n_neighbors', type=int, default=5, 
                    help='Number of neighboring samples to use for KNN imputation.')
parser.add_argument('--test_module', action="store_true", default=False, 
                    help='Add random noise for test module')
parser.add_argument('--noise_ratio', type=float, default=0.05, help='randomly insert NA noises with this value')


class DataImputer():
    def __init__(self, data_path, header):
        self.dataset = pd.read_csv(data_path, header = None)
        
    def add_random_noise(self, noise_ratio):  
        dataset_na = self.dataset.copy()
        
        # select only numeric columns to apply the missingness to
        cols_list = dataset_na.select_dtypes('number').columns.tolist()

        # randomly insert NA values
        for col in self.dataset[cols_list]:
            dataset_na.loc[self.dataset.sample(frac=noise_ratio).index, col] = np.nan
            
        return dataset_na
    
    def SimpleImputer(self, strategy, fill_value):
        """ sklearn simple imputer. """
        # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html         
        imputer = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=fill_value)
        imputed_dataset = imputer.fit_transform(self.dataset)
        imputed_dataset = pd.DataFrame(imputed_dataset, columns=self.dataset.columns)
        
        return imputed_dataset
    
    def KNNImputer(self, n_neighbors):
        """ sklearn KNN imputer. """
        # https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_dataset = imputer.fit_transform(self.dataset)
        imputed_dataset = pd.DataFrame(imputed_dataset, columns=self.dataset.columns)
        
        return imputed_dataset
    
    def MICEImputer(self, initial_strategy):
        """ sklearn MICE imputer. """
        # https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn-impute-iterativeimputer
        imputer = IterativeImputer(random_state=0, initial_strategy=initial_strategy, sample_posterior=True)
        imputed_dataset = imputer.fit_transform(self.dataset)
        imputed_dataset = pd.DataFrame(imputed_dataset, columns=self.dataset.columns)
        
        return imputed_dataset

def main(args):
    dataImputer = DataImputer(args.data_path, args.header)
    
    if args.test_module:
        # Add Noise
        print('Add random noise with NA values for testing the module.')
        dataImputer.dataset = dataImputer.add_random_noise(args.noise_ratio)
        dataImputer.dataset.to_csv('./dataset/noisy_dataset.csv', mode='w', index=False, header=False)
    
    if args.option == 'simple':
        assert(args.strategy is not None)
        print(f'Processing simple imputation using the {args.strategy}')
        imputed_dataset = dataImputer.SimpleImputer(strategy=args.strategy, fill_value=args.fill_value)
    elif args.option == 'knn':
        assert(args.n_neighbors > 0)
        print(f'Processing KNN imputation')
        imputed_dataset = dataImputer.KNNImputer(n_neighbors=args.n_neighbors)
    elif args.option == 'mice':
        print(f'Processing MICE imputation')
        imputed_dataset = dataImputer.MICEImputer(initial_strategy=args.strategy)
        
    imputed_dataset.to_csv(args.output_path, mode='w', index=False, header=False)
    print(f'Done {args.option} imputation')
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)