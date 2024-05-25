import numpy as np
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss

from TabularUtilities import TabularMLP
from TrainUtilities import TripletDataset, train_triplets, save_model, evaluate_nddg
from PreprocessingUtilities import sample_manager

#Required Paths
current_directory = os.getcwd()
images_path='../data/images/'
csvs_path ='../data/csvs/'
pickle_path = current_directory + '/../data/pickles/'
path_save = '../bin/'
favorite_image_info = csvs_path + 'favorite_image_info.csv'
patient_info = csvs_path + 'patient_info.csv'
patient_images_info = csvs_path + 'patient_images.csv'
catalogue_info = csvs_path + 'catalogue_info.csv'
catalogue_user_info = csvs_path + 'catalogue_user_info.csv'

# Configs
np.random.seed(10)
torch.manual_seed(10)
device = "cuda:0" # "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
lr=0.0001
num_epochs=100
batch_size=512
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

# Implemented Model
models = {
    "Tabular-MLP": TabularMLP(5, 16, 5)
}

for model_name, model in models.items():
    # # Define Dataset & Dataloaders & Optimization Parameters
    train_dataset = TripletDataset('', QNS_list_tabular_train, transform=model.get_transform())
    test_dataset  = TripletDataset('', QNS_list_tabular_test,  transform=model.get_transform())
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # later it should bea turned on ...
    test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    criterion     = TripletMarginLoss(margin=margin, p=2)
    optimizer     = optim.Adam(model.parameters(), lr=lr)

    print(f'Training {model_name}...')
    model, _, _ = train_triplets(model, train_loader, test_loader, QNS_list_tabular_train, QNS_list_tabular_test, optimizer, criterion, num_epochs=num_epochs, model_name=model_name, device=device, path_save=path_save)

    print(f'Saving {model_name}...')
    save_model(model, f'{path_save}{model_name}/Finale.pl')
    print(f'Done {model_name}!')