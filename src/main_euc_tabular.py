import numpy as np
import torch
import os
from PreprocessingUtilities import sample_manager
from TabularUtilities import euclidian_optimizer, euclidian_evaluate
import sys

#Required Paths
current_directory = os.getcwd()
images_path='../data/images/'
csvs_path ='../data/csvs/'
pickle_path = current_directory + '/../data/pickles/'
path_save = '../bin/Tabular_Eucl'
favorite_image_info = csvs_path + 'favorite_image_info.csv'
patient_info = csvs_path + 'patient_info.csv'
patient_images_info = csvs_path + 'patient_images.csv'
catalogue_info = csvs_path + 'catalogue_info.csv'
catalogue_user_info = csvs_path + 'catalogue_user_info.csv'

# Configs
np.random.seed(10)
torch.manual_seed(10)
device = "cpu"
print(f"Using device: {device}")
lr=0.01
num_epochs = 200
margin = 0.0001
split_ratio=0.8
catalogue_type = 'E'
doctor_code=-1 # 39 57 36 -1

# Preprocessing
QNS_list_image_train, QNS_list_image_test, QNS_list_tabular_train, QNS_list_tabular_test = \
sample_manager(images_path, pickle_path, catalogue_info, catalogue_user_info, 
patient_info, favorite_image_info, patient_images_info, catalogue_type=catalogue_type, doctor_code=doctor_code, split_ratio=split_ratio, default=False)

# for q in QNS_list_tabular_train:
#     q.show_summary(str=False)

# for q in QNS_list_tabular_test:
#     q.show_summary(str=False)

models={'Vector': {'mat_stat': False, 'dim_stat': 0},
        'Matrix': {'mat_stat': True, 'dim_stat': 0},
        'Semi-Positive': {'mat_stat': False, 'dim_stat': 10}
}

os.makedirs(f'{path_save}', exist_ok=True)
for model_name, model_info  in models.items():
    is_mat = model_info['mat_stat']
    dim = model_info['dim_stat']

    with open(f'{path_save}/Train_info_{model_name}_not_normalized.log', 'a') as f:
        original_stdout = sys.stdout  # Save the original stdout
        sys.stdout = f
        print(f'Model: {model_name} is_mat is: {is_mat} dim is: {dim}')
        optimized_weights = euclidian_optimizer(QNS_list_tabular_train, QNS_list_tabular_test, is_mat=is_mat, dim=dim, lr=lr, num_epochs=num_epochs)
        print(optimized_weights)

    sys.stdout = original_stdout

print('Done!')