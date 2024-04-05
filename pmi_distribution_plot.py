import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Customize matplotlib settings
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 13
plt.rcParams["figure.figsize"] = (3.54, 3.54*0.8)
# plt.rcParams["figure.figsize"] = (5, 3)

root_dir = 'train-test-data/'
out_dir = 'figures/box-plot/'
subject_disj_fold_list = {'nir': [3, 10], 'rgb': [9, 8], 'multispectral': [6, 10]}
entire_10_fold_list = {'nir': [1, 10], 'rgb': [1, 10], 'multispectral': [1, 7]}
performance = ["least", "best"]

# List of subdirectories
subdirs = os.listdir(root_dir)
w = 50  # Increase bin width

for subdir in subdirs:
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        sub_subdirs = os.listdir(subdir_path)
        for sub_subdir in sub_subdirs:
            sub_subdir_path = os.path.join(subdir_path, sub_subdir)
            if os.path.isdir(sub_subdir_path):
                if sub_subdir_path.split("/")[-2] != "dataset-disjoint":
                    print(sub_subdir_path)
                    key = sub_subdir_path.split('/')[-1]

                    if sub_subdir_path.split("/")[-2] == "10-fold":
                        fold_dict = entire_10_fold_list[key]
                    else:
                        fold_dict = subject_disj_fold_list[key]

                    for i in range(len(fold_dict)):
                        # Paths to train and test data files
                        train_data_file = glob.glob(sub_subdir_path + f"/fold_{fold_dict[i]}_train.*")[0]
                        test_data_file = glob.glob(sub_subdir_path + f"/fold_{fold_dict[i]}_test.*")[0]

                        # Figure name for saving
                        figname = out_dir + sub_subdir_path.split("/")[-2] + "-" + key + "-" + performance[i] + "-box-plot.pdf"

                        # Read train and test data
                        train_data = pd.read_csv(train_data_file)
                        test_data = pd.read_csv(test_data_file)

                        # Filter data
                        train_data = train_data[train_data["pmi"] < 1700]
                        test_data = test_data[test_data["pmi"] < 1700]

                        print(train_data_file)
                        print(test_data_file)
                        print(figname)

                        # Create bins
                        trainMin = min(train_data['pmi'])
                        trainMax = max(train_data['pmi'])
                        testMin = min(test_data['pmi'])
                        testMax = max(test_data['pmi'])

                        if trainMin < testMin:
                            lower = trainMin
                        else:
                            lower = testMin

                        if trainMax > testMax:
                            upper = trainMax
                        else:
                            upper = testMax

                        bins = np.arange(lower, upper + w, w)

                        # Create box plot for train and test data
                        plt.figure()
                        sns.histplot(train_data['pmi'], label='Train', bins=bins, stat='probability', common_norm=False)
                        sns.histplot(test_data['pmi'], label='Test', bins=bins, stat='probability', common_norm=False)
                        plt.xlabel('PMI')
                        plt.ylabel('Probability')
                        plt.legend(loc='upper right')
                        plt.yscale('log')
                        plt.grid()
                        plt.savefig(figname, format='pdf', dpi=600)
                        plt.clf()
                    print()

                else:
                    key = sub_subdir_path.split('/')[-1]
                    figname = out_dir + sub_subdir_path.split("/")[-2] + "-" + key + "-box-plot.pdf"
                    warsaw_file = glob.glob(sub_subdir_path + f"/warsaw*.txt")[0]
                    nij_file = glob.glob(sub_subdir_path + f"/nij*.txt")[0]

                    warsaw_data = pd.read_csv(warsaw_file)
                    nij_data = pd.read_csv(nij_file)

                    warsaw_data = warsaw_data[warsaw_data["pmi"] < 1700]
                    nij_data = nij_data[nij_data["pmi"] < 1700]

                    print(warsaw_file)
                    print(nij_file)
                    print(figname)

                    # Create bins
                    trainMin = min(nij_data['pmi'])
                    trainMax = max(nij_data['pmi'])
                    testMin = min(warsaw_data['pmi'])
                    testMax = max(warsaw_data['pmi'])

                    if trainMin < testMin:
                        lower = trainMin
                    else:
                        lower = testMin

                    if trainMax > testMax:
                        upper = trainMax
                    else:
                        upper = testMax

                    bins = np.arange(lower, upper + w, w)

                    # Create box plot for train and test data
                    plt.figure()
                    sns.histplot(nij_data['pmi'], label='NIJ', bins=bins, stat='probability', common_norm=False)
                    sns.histplot(warsaw_data['pmi'], label='Warsaw', bins=bins, stat='probability', common_norm=False)
                    plt.xlabel('PMI')
                    plt.ylabel('Probability')
                    plt.legend(loc='upper right')
                    plt.yscale('log')
                    plt.grid()
                    plt.savefig(figname, format='pdf', dpi=600)
                    plt.clf()

                    print()
