import glob
import pandas as pd
import random
import os

input_dir = './train-test-data-list/'
files = glob.glob(os.path.join(input_dir, "*.txt"))

for file in files:
    print(file)
    df = pd.read_csv(file)
    grouped_data = df.groupby('subject_id')
    subject_groups = list(grouped_data.groups.keys())
    random.shuffle(subject_groups)

    num_folds = 10
    folds = [subject_groups[i::num_folds] for i in range(num_folds)]

    output_dir = file.split('/')[-1].split('-')[1]
    os.makedirs(output_dir, exist_ok=True)

    for i, fold in enumerate(folds):
        test_subjects = fold
        temp_train_subjects = [subject for subject in subject_groups if subject not in fold]

        # Create train and test sets based on selected subjects
        temp_train_set = df[df['subject_id'].isin(temp_train_subjects)]
        test_set = df[df['subject_id'].isin(test_subjects)]

        # Split train and validation set
        temp_train_grouped_data = temp_train_set.groupby('subject_id')
        temp_train_subject_groups = list(temp_train_grouped_data.groups.keys())
        random.shuffle(temp_train_subject_groups)

        # perform 10-fold on training set and take first fold for validation
        train_folds = [temp_train_subject_groups[i::num_folds] for i in range(num_folds)]
        val_subjects = train_folds[random.randint(0, 9)]
        train_subjects = [subject for subject in temp_train_subject_groups if subject not in val_subjects]

        # Create train and validation sets based on selected subjects
        train_set = temp_train_set[temp_train_set['subject_id'].isin(train_subjects)]
        val_set = temp_train_set[ temp_train_set['subject_id'].isin(val_subjects)]

        # Save the sets to CSV files
        train_set.to_csv(os.path.join(output_dir, f'fold_{i + 1}_train.csv'), index=False)
        val_set.to_csv(os.path.join(output_dir, f'fold_{i + 1}_validation.csv'), index=False)
        test_set.to_csv(os.path.join(output_dir, f'fold_{i + 1}_test.csv'), index=False)

        # Add this code after creating train_set, val_set, and test_set in your existing loop
        subject_intersection_train_val = set(train_set['subject_id']).intersection(set(val_set['subject_id']))
        subject_intersection_train_test = set(train_set['subject_id']).intersection(set(test_set['subject_id']))
        subject_intersection_val_test = set(val_set['subject_id']).intersection(set(test_set['subject_id']))

        if subject_intersection_train_val or subject_intersection_train_test or subject_intersection_val_test:
            print(f"Subjects are not unique across sets in {output_dir}")
        else:
            print(f"Subjects are unique across sets in {output_dir}")


