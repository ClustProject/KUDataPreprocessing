"""Seasonal Trend Decomposition and Prediction module for Numeric dataset.

"""

import pandas as pd
from dtw import dtw
import matplotlib.pyplot as plt
import argparse
from tslearn import metrics


parser = argparse.ArgumentParser(description='Temporal One-class Anomaly Detection')
parser.add_argument('--data_path', type=str, default='./dataset/power_voltage.csv')
parser.add_argument('--dtw_output_path', type=str, default='./dataset/synchronized_data/synchronized_dtw.csv')
parser.add_argument('--plot_output_path', type=str, default='./dataset/synchronized_data')
parser.add_argument('--soft_dtw_output_path', type=str, default='./dataset/synchronized_data/synchronized_softdtw.csv')
parser.add_argument('--option',  type=str, default='dtw', help='synchronization method')
parser.add_argument('--gamma',  type=float, default=0.1, help='gamma for soft dtw')
parser.add_argument('--distance',  type=float, default= 2, help='p-norm for distacne')
parser.add_argument('--plot',  type=bool, default= True, help='plot for visualization')

class Sync():
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path)
        self.Time = self.dataset.iloc[:,0]
        self.X = self.dataset.iloc[:,1].fillna(0).values
        self.Y = self.dataset.iloc[:,2].fillna(0).values

    def dtw_sync(self, distance):

        distance, path = dtw(self.X, self.Y, dist=distance)

        result = []

        for i in range(0, len(path)):
            result.append([self.Time[path[i][0]], self.X[path[i][0]], self.Y[path[i][1]]])

        result_df = pd.DataFrame(data=result, columns=['Time', 'X', 'Y']).dropna()
        result_df = result_df.drop_duplicates(subset=['Time'])
        result_df = result_df.sort_values(by='Time')
        result_df = result_df.reset_index(drop=True)

        return result_df

    def soft_dtw_sync(self, gamma):

        path, sim = metrics.soft_dtw_alignment(self.X, self.Y, gamma=gamma)

        result = []

        for i in range(0, len(path)):
            result.append([self.Time[path[i][0]], self.X[path[i][0]], self.Y[path[i][1]]])

        result_df = pd.DataFrame(data=result, columns=['Time', 'X', 'Y']).dropna()
        result_df = result_df.drop_duplicates(subset=['Time'])
        result_df = result_df.sort_values(by='Time')
        result_df = result_df.reset_index(drop=True)

        return result_df


    def plot_data(self):

        return self.X, self.Y


def main(args):
    DTW_sync = Sync(args.data_path)

    if args.option == 'dtw':
        print(f'Processing synchronization using the {args.option}')
        result_df = DTW_sync.dtw_sync(distance=args.distance)
        result_df.to_csv(args.dtw_output_path, index=False)

    elif args.option == 'soft_dtw':
        print(f'Processing synchronization using the {args.option}')
        result_df = DTW_sync.dtw_sync(distance=args.gamma)
        result_df.to_csv(args.soft_dtw_output_path, index=False)

    if args.plot == True:

        X, Y = DTW_sync.plot_data()
        plt.plot(X[:100])
        plt.plot(Y[:100])
        plt.title('Original data (0~100 timestep)')
        plt.savefig(args.plot_output_path + '/Original plot.png')
        plt.clf()

        plt.plot(result_df['X'][0:100])
        plt.plot(result_df['Y'][0:100])
        plt.title(f'Synchronized by {args.option} (0~100 timestep)')
        plt.savefig(args.plot_output_path + '/Synchronized plot.png')


    print(f'Done {args.option} synchronization')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)