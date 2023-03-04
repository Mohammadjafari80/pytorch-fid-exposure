import csv
from constants import dataset_labels

import os
import csv

dataset_names = ['mnist', 'fashion', 'cifar10', 'cifar100']
exposure_datasets = ['mnist', 'fashion', 'cifar10', 'cifar100', 'svhn', 'adaptive']

for dataset_name in dataset_names:
    num_classes = len(dataset_labels[dataset_name])
    normal_classes = range(num_classes)
    fid_results = {}

    for normal_class in normal_classes:
        log_file_pattern = f'FID-{dataset_name}-{normal_class:02d}-{dataset_labels[dataset_name][normal_class]}-{{}}_logs.txt'
        class_dir = os.path.join('./Scores/', f'normal-{dataset_name}', f'normal-class-{normal_class:02d}-{dataset_labels[dataset_name][normal_class]}')

        for exposure_dataset in exposure_datasets:
            results_file = os.path.join(class_dir, f'exposure-{exposure_dataset}', 'run_00', log_file_pattern.format(exposure_dataset))

            with open(results_file, 'r') as f:
                for line in f:
                    print(line)
                    if 'FID' in line:
                        fid_value = float(line.split(': ')[-1].strip())
                        if normal_class not in fid_results:
                            fid_results[normal_class] = {}
                        fid_results[normal_class][exposure_dataset] = fid_value

    # Write the fid_results to a CSV file
    # Write the fid_results to a CSV file
    results_dir = './Results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    csv_filename = os.path.join(results_dir, f'fid_results-{dataset_name}.csv')
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Normal Class'] + exposure_datasets)
        for normal_class in normal_classes:
            row = [dataset_labels[dataset_name][normal_class]]
            for exposure_dataset in exposure_datasets:
                if normal_class in fid_results and exposure_dataset in fid_results[normal_class]:
                    row.append(fid_results[normal_class][exposure_dataset])
                else:
                    row.append('')
            writer.writerow(row)