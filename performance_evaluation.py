import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Customize matplotlib settings
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 15
plt.rcParams["figure.figsize"] = (3.54, 3.54*0.8)

# Data directory
root_dirs = ['10-fold-results/', 'subject-disj-10-fold-results/', 'warsaw-as-testset/', 'nij-as-testset']
sub_dirs = ['nir/', 'rgb/', 'multispectral/']
model_list = ['densenet', 'inception', 'resnet', 'vgg']
out_dir = 'figures/reg-plot/'

colors = sns.color_palette("deep")

for root_dir in root_dirs:
    for subdir in sub_dirs:
        subdir_path = os.path.join(root_dir, subdir)
        print(subdir_path)
        for model_name in model_list:
            files = glob.glob(os.path.join(subdir_path, f"{model_name}*.txt"))

            data_list = []
            rmse_list = []
            mae_list = []
            for file in files:
                data = pd.read_csv(file)
                rmse = math.sqrt(mean_squared_error(data['Actual'], data['Predicted']))
                mae = mean_absolute_error(data['Actual'], data['Predicted'])

                data_list.append(data)
                rmse_list.append(rmse)
                mae_list.append(mae)

            if len(rmse_list) > 1:
                avg_rmse = statistics.mean(rmse_list)
                rmse_stdev = statistics.stdev(rmse_list)
            else:
                avg_rmse = rmse
                rmse_stdev = 0

            if len(mae_list) > 1:
                avg_mae = statistics.mean(mae_list)
                mae_stdev = statistics.stdev(mae_list)
            else:
                avg_mae = mae
                mae_stdev = 0

            print(f"Model: {model_name} Avg RMSE: {avg_rmse:.2f} RMSE Std: {rmse_stdev:.2f} Avg MAE: {avg_mae:.2f} MAE Std: {mae_stdev:.2f}")

            if len(data_list) > 1:
                combined_data = pd.concat(data_list, ignore_index=True)
            else:
                combined_data = data

            figname = f"{out_dir}{root_dir.split('-')[0]}_{subdir.split('/')[0]}_{model_name}.pdf"

            # Create a scatter plot for the current fold with RMSE and MAE in a box
            plt.scatter(combined_data['Actual'], combined_data['Predicted'], s=5, color='royalblue', label='Predicted')
            plt.plot(combined_data['Actual'], combined_data['Actual'], color='darkorange', label='Actual', linestyle='dashed', markersize=5)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.legend()
            plt.grid(True)
            plt.savefig(figname, format='pdf', dpi=600)
            plt.clf()

        print()
