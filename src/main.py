import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import string
from random import sample
from scipy.spatial.distance import euclidean
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from itertools import combinations
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss
from transformers import AutoImageProcessor, ViTImageProcessor, ViTModel, DeiTImageProcessor, DeiTModel, BeitImageProcessor, BeitModel, Dinov2Model, ConvNextV2Model

from transformers import  ViTConfig, AutoModel
#, SwinModel, ConvNextModel, CaitModel, SwinFeatureExtractor, ConvNextImageProcessor, ConvNextConfig, CaitFeatureExtractor

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class TripletDataset(Dataset):
    def __init__(self, path, QNS_list, transform):
        self.transform = transform
        self.path = path
        # precompute all combination of the triplets
        self.triplets = []
        for qns_element in QNS_list:
            for pair in combinations(range(qns_element.neighbor_count), 2):
                self.triplets.append((qns_element.query_vector, qns_element.
                neighbor_vectors[pair[0]], qns_element.neighbor_vectors[pair[1]]))

    def print_triplets(self):
        for i in self.triplets:
            print(i)

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, index):
        query, pos, neg = self.triplets[index]
        return {
            'query': self.transform(query),  # Assuming transform outputs a dict with 'pixel_values'
            'pos': self.transform(pos),
            'neg': self.transform(neg)
        }

def save_model(model, filepath):
    """
    Save the model's state dictionary to a file.

    Args:
    - model: The PyTorch model to save.
    - filepath: The path where the model will be saved.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(model, filepath, device='cpu'):
    """
    Load the model's state dictionary from a file.

    Args:
    - model: The PyTorch model to load the state dictionary into.
    - filepath: The path from where the model will be loaded.
    - device: The device where the model should be loaded ('cpu' or 'cuda:0').

    Returns:
    - model: The loaded PyTorch model.
    """
    # Load the model's state dictionary
    model.load_state_dict(torch.load(filepath, map_location=device))
    
    # Set the model to evaluation mode
    model.eval()
    
    print(f"Model loaded from {filepath}")
    return model

def train_triplets(model, train_loader, test_loader, optimizer, criterion, num_epochs, device='cpu', save_path='../tmp/'):
    model.to(device)
    model.train()  # Set model to training mode
    best_acc = float('-inf')  
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:

            # Zero the parameter gradients
            optimizer.zero_grad()  

            queries = data['query'].to(device)
            positives = data['pos'].to(device)
            negatives = data['neg'].to(device)
            
            # Forward pass to get outputs and calculate loss
            anchor_embeddings = model(queries)
            pos_embeddings    = model(positives)
            neg_embeddings    = model(negatives)
            loss = criterion(anchor_embeddings, pos_embeddings, neg_embeddings)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss = running_loss + loss.item() * queries.size(0)
        
        # Evaluation
        epoch_loss = running_loss # / len(train_loader.dataset)
        train_acc = evaluate_triplets(model, train_loader, device)
        test_acc  = evaluate_triplets(model, test_loader, device)

        current_time = datetime.now()
        print(f'[{current_time.strftime("%Y-%m-%d %H:%M:%S")}] Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train-Acc: {train_acc:.5f}, Test-Acc: {test_acc:.5f}')
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train-Acc: {train_acc:.5f}, Test-Acc: {test_acc:.5f}')

        # Saving
        if test_acc > best_acc:
            best_acc = test_acc
            # Save the best model
            torch.save(model.state_dict(), f'{save_path}{current_time.strftime("%Y-%m-%d %H_%M_%S")}-{best_acc:.5f}.pl')
            print(f"New best model saved with accuracy: {best_acc:.5f}")

    print('Finished Training')
    return model, epoch_loss, epoch

def evaluate_triplets(model, data_loader, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    total_triplets = 0
    correct_predictions = 0
    total_pos_distance = 0.0
    total_neg_distance = 0.0
    
    with torch.no_grad():  # No gradients needed
        for data in data_loader:

            queries = data['query'].to(device)
            positives = data['pos'].to(device)
            negatives = data['neg'].to(device)
            
            # Get embeddings for each part of the triplet
            anchor_embeddings = model(queries)
            pos_embeddings    = model(positives)
            neg_embeddings    = model(negatives)
            
            # Compute distances
            pos_distances = torch.norm(anchor_embeddings - pos_embeddings, p=2, dim=1)
            neg_distances = torch.norm(anchor_embeddings - neg_embeddings, p=2, dim=1)
            
            # Update total distances
            total_pos_distance = total_pos_distance + pos_distances.sum().item()
            total_neg_distance = total_neg_distance + neg_distances.sum().item()
            
            # Count correct predictions (positive distance should be less than negative distance)
            correct_predictions = correct_predictions + (pos_distances < neg_distances).sum().item()
            total_triplets = total_triplets + queries.size(0)

    # Calculate average distances
    #avg_pos_distance = total_pos_distance / total_triplets
    #avg_neg_distance = total_neg_distance / total_triplets
    
    # Calculate accuracy
    accuracy = correct_predictions / total_triplets

    return accuracy    

def preprocess_single_sample(image_path, transform):
    x = Image.open(image_path).convert('RGB')
    x = transform(x)['pixel_values'].unsqueeze(0)
    return x

def evaluate_nddg(QNS_list, model, transform, device='cpu'):
    final_order = []
    model.eval()
    with torch.no_grad():  # No need to track gradients during evaluation
        for q_element in QNS_list:
            fss = []
            # Load and transform the query image
            query_tensor = transform(q_element.query_vector).unsqueeze(0).to(device)
            vec_ref = model(query_tensor)

            for neighbor_path in q_element.neighbor_vectors:
                # Load and transform the neighbor image
                neighbor_tensor = transform(neighbor_path).unsqueeze(0).to(device)
                vec_i = model(neighbor_tensor)

                distf = torch.norm(vec_ref - vec_i)
                fss.append(distf.item())
            final_order.append(fss) 

    model_acc = 100 * np.mean(test_ndcg(final_order))
    return model_acc, final_order

class Google_Base_Patch16_224(nn.Module):
    def __init__(self):
        super(google_base_patch_16_224, self).__init__()
        # Load the pre-trained deit_tiny_patch16_224 ViT model
        self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        # Assuming the model outputs the last_hidden_state directly
        featureVec = outputs.last_hidden_state[:, 0, :]  # Use outputs.last_hidden_state if no pooling
        return featureVec

class DeiT_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = DeiTImageProcessor.from_pretrained('facebook/deit-base-patch16-224')
        self.model = DeiTModel.from_pretrained('facebook/deit-base-patch16-224')

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token's embeddings

class Beit_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
        self.model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224')

    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]

class DinoV2_Base_Patch16_224(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = Dinov2Model.from_pretrained('facebook/dinov2-base')
        
    def get_transform(self):
        def transform(image_path):
            image = Image.open(image_path).convert('RGB')
            processed = self.feature_extractor(images=image, return_tensors="pt")
            return processed['pixel_values'].squeeze(0)
        return transform
    
    def forward(self, input):
        outputs = self.model(input)
        return outputs.last_hidden_state[:, 0, :]




# class ConvNextV2_Base_Patch16_224(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor =  AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
#         self.model = ConvNextV2Model.from_pretrained("facebook/convnextv2-tiny-1k-224")
        
#     def get_transform(self):
#         def transform(image_path):
#             image = Image.open(image_path).convert('RGB')
#             processed = self.feature_extractor(images=image, return_tensors="pt")
#             return processed['pixel_values'].squeeze(0)
#         return transform
    
#     def forward(self, input):
#         outputs = self.model(input)
#         return outputs.last_hidden_state[:, 0, :]

# class Cait_base_patch_16_224(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor = CaitIImageProcessor.from_pretrained('facebook/cait-base-patch16-224')
#         self.model = CaitModel.from_pretrained('facebook/cait-base-patch16-224')

#     def get_transform(self):
#         def transform(image_path):
#             image = Image.open(image_path).convert('RGB')
#             processed = self.feature_extractor(images=image, return_tensors="pt")
#             return processed['pixel_values'].squeeze(0)
#         return transform
    
#     def forward(self, input):
#         outputs = self.model(input)
#         return outputs.last_hidden_state[:, 0, :]

# class BaseNoPoolViTModel(ViTModel):
#     def __init__(self, config):
#         super().__init__(config)
#         # Deactivate the pooler directly if it exists
#         self.pooler = None

#     def forward(
#         self,
#         pixel_values,
#         head_mask=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         outputs = super().forward(
#             pixel_values=pixel_values,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         # Use last hidden state since pooler is deactivated
#         last_hidden_state = outputs.last_hidden_state
#         return last_hidden_state

# class NoPoolModifiedViT(nn.Module):
#     def __init__(self):
#         super(NoPoolModifiedViT, self).__init__()
#         # Configure and initialize the custom ViT model
#         config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
#         self.model = BaseNoPoolViTModel(config)
#         self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
#     def get_transform(self):
#         def transform(image_path):
#             image = Image.open(image_path).convert('RGB')
#             processed = self.feature_extractor(images=image, return_tensors="pt")
#             return processed['pixel_values'].squeeze(0)
#         return transform
    
#     def forward(self, input):
#         outputs = self.model(input)
#         # Assuming the model outputs the last_hidden_state directly
#         featureVec = outputs[:, 0, :]  
#         return featureVec

# class google_base_patch_16_224_MLP(nn.Module):
#     def __init__(self):
#         super(google_base_patch_16_224_MLP, self).__init__()
#         # Load the pre-trained deit_tiny_patch16_224 ViT model
#         self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
#         self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    
#     # Define MLP layers
#         self.fc1 = nn.Linear(768, 512)  # First MLP layer (change 768 to your feature size)
#         self.relu1 = nn.ReLU()          # ReLU activation
#         self.fc2 = nn.Linear(512, 256)  # Second MLP layer
#         self.relu2 = nn.ReLU()          # ReLU activation

#     def get_transform(self):
#         def transform(image_path):
#             image = Image.open(image_path).convert('RGB')
#             processed = self.feature_extractor(images=image, return_tensors="pt")
#             return processed['pixel_values'].squeeze(0)
#         return transform
    
#     def forward(self, input):
#         outputs = self.model(input)
#         # Assuming the model outputs the last_hidden_state directly
#         featureVec = outputs.last_hidden_state[:, 0, :]  # Use outputs.last_hidden_state if no pooling
#         x = self.fc1(featureVec)
#         x = self.relu1(x)
#         x = self.fc2(x)
#         featureVec = self.relu2(x)
#         return featureVec

# class Swin_Base_Patch4_Window7_224(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor = SwinFeatureExtractor.from_pretrained('microsoft/swin-base-patch4-window7-224')
#         self.model = SwinModel.from_pretrained('microsoft/swin-base-patch4-window7-224')

#     def get_transform(self):
#         def transform(image_path):
#             image = Image.open(image_path).convert('RGB')
#             processed = self.feature_extractor(images=image, return_tensors="pt")
#             return processed['pixel_values'].squeeze(0)
#         return transform
    
#     def forward(self, input):
#         outputs = self.model(input)
#         return outputs.last_hidden_state[:, 0, :]

# class ConvNext_base_patch_16_224(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # This will automatically load the necessary config along with the model weights.
#         self.feature_extractor = ConvNextImageProcessor.from_pretrained('facebook/convnext-base-224')
#         self.model = ConvNextModel.from_pretrained('facebook/convnext-base-224')

#     def get_transform(self):
#         def transform(image_path):
#             image = Image.open(image_path).convert('RGB')
#             processed = self.feature_extractor(images=image, return_tensors="pt")
#             return processed['pixel_values'].squeeze(0)
#         return transform
    
#     def forward(self, input):
#         outputs = self.model(input)
#         return outputs.last_hidden_state[:, 0, :]


#Returns normalized discounted Cumulative gain
def test_ndcg(distances):       
  res = np.zeros(len(distances))
  for i in range(len(distances)):
    dcg_aux = 0
    idcg_aux = 0
    ndcg = 0
    dist = distances[i]
    sorted_indexes = np.argsort(dist)
    new_array = np.argsort(sorted_indexes) #Contains the position of each patient in an ordered list
    for z in range(len(dist)):      
      dcg_aux += (len(dist)-z) / (np.log(new_array[z]+2)/np.log(2))
      idcg_aux += (len(dist)-z) / (np.log(z+2)/np.log(2))

    res[i]= dcg_aux/idcg_aux

  return res

def is_valid_date(date_str):
    try:
        if isinstance(date_str, str):
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        else:
            return False
    except ValueError:
        return False

# Function: Converts given date to age
def convert_birthday_to_age(birthday_date):
    birthday_date = datetime.strptime(birthday_date, "%Y-%m-%d").date()
    today = datetime.today()
    age = today.year - birthday_date.year - ((today.month, today.day) < (birthday_date.month, birthday_date.day))
    return age

# Function: Convert bra band size to under-bust (interval midpoint)
def convert_brasize_to_underbust(bra_band):

    """
    International / European / Japan / South Korea
    Measurements in centimeters (cm)
    """

    # Build the dict
    bra_band_dict = dict()
    for b in range(55, 155, 5):
        bra_band_dict[f'{b}'] = pd.Interval(left=b, right=b+5, closed='left')

    # And compute the under_bust
    try:
        under_bust = bra_band_dict[f'{bra_band}'].mid
    except:
        under_bust = None

    return under_bust

# Function: Convert bra cup to difference between over and under-bust (interval midpoint)
def convert_bracup_to_oubustdiff(bra_cup):

    """
    International / Europe / France / Spain / Belgium / Italy
    Measurements in centimetres (cm)
    """

    # Get the possible bra cups
    iefebi_bra_cups = ['AA']
    iefebi_bra_cups += list(string.ascii_uppercase)[0:20]
    iefebi_bra_cups.remove('Q')

    # Create a dictionary of bra cups and intervals of differences
    iefebi_bra_cups_dict = dict()
    for c, b in zip(iefebi_bra_cups, range(10, 50, 2)):
        iefebi_bra_cups_dict[c] = pd.Interval(left=b, right=b+2, closed='left')

    # And compute the bust_ou_diff
    try:
        bust_ou_diff_out  = iefebi_bra_cups_dict[bra_cup.upper()].mid
    except:
        bust_ou_diff_out = None

    return bust_ou_diff_out

class QNS_structure:
    def __init__(self, query_vector=None, neighbor_vectors=[], scores=None, cmpfunc=euclidean, query_vector_id=-1, neighbor_vectors_ids=[]):
        
        self.set_query_vector(query_vector, query_vector_id)
        self.set_neighbor_vectors(neighbor_vectors, neighbor_vectors_ids)
        self.set_score_vector(scores)
        self.dist_func = cmpfunc
        
    def set_query_vector(self, query_vector,query_vector_id=-1):
        self.query_vector = query_vector #+ 1e-15 * np.ones(query_vector.Shape, np.float64)
        self.query_vector_id = query_vector_id

    def append_to_vector(self, vec1, vec2):
        vec1 = np.append(vec1, vec2)
        return vec1

    def set_score_vector(self, score_vector):
        self.score_vector = score_vector
    
    def set_neighbor_vectors(self, neighbor_vectors, neighbor_vectors_ids):
        if neighbor_vectors == []:
            self.neighbor_count = 0
            self.neighbor_vectors = []
            self.neighbor_vectors_id = []
        else:
            self.neighbor_count = len(neighbor_vectors)
            self.neighbor_vectors = neighbor_vectors
            self.set_neighbor_vectors_ids = neighbor_vectors_ids

    def calculate_expert_score(self):
        if self.neighbor_count == 0:
           return
        self.score_vector = np.arange(self.neighbor_count, dtype=np.float64)

    def show_neighbor_vectors(self):
        for idx, neighbor in enumerate(self.neighbor_vectors):
            print(f'Neighbor PID: {self.neighbor_vectors_id[idx]} NID: {idx} Vec: {neighbor}')

    def show_query_vector(self):
        print(f'Query => PID: {self.query_vector_id} | Vec: {self.query_vector}')

    def show_score_vector(self):
        print('Score-Vector: ', self.score_vector)

    def show_summary(self,num_flag=False):

        if num_flag:
            init_dist = self.calculate_initial_distance ()
            init_score = self.calculate_initial_score ()
        
        self.show_query_vector()
        for idx, neighbor in enumerate(self.neighbor_vectors):
            print(f'Neighbor NID:{idx:02} => Value: [ ', end='')
            for n in neighbor:
                print(f'{n}', end='')
            if num_flag:
                print(f'Init Dist: {init_dist[idx]:06.3f}  | Dist-Based Score: {init_score[idx]:02d}')
            print(f' | Expert Score: {self.score_vector[idx]:02.1f}')

        if num_flag:
            print("Initial Sort: ", self.calculate_initial_sort())
        print("Neighbor PID: ", self.neighbor_vectors_id)
        print("Expert  Sort: ", self.calculate_expert_sort())
        print("----------------------------------")
        
    def calculate_initial_distance (self):
        initial_distance = [self.dist_func(self.query_vector, neighbor) for neighbor in self.neighbor_vectors]
        if self.dist_func is not euclidean:
            initial_distance = 1.0 - initial_distance

        return initial_distance  

    def calculate_initial_score (self):
        sorted_indices = np.argsort(np.array(self.calculate_initial_distance())) # Use argsort to get the indices that would sort the array
        initial_score  = np.zeros_like(sorted_indices)
       
        for score, index in enumerate(sorted_indices):
            initial_score[index] = score

        return initial_score

    def calculate_initial_sort(self):
        initial_sort  = np.argsort(self.calculate_initial_distance())
        return initial_sort

    def calculate_expert_sort(self):
        if self.score_vector is not None:
            expert_sort  = np.argsort(self.score_vector)
        return expert_sort

    def add_neighbor_vector(self, neighbor_vector, neighbor_vector_id=-1):
        self.neighbor_vectors.append(neighbor_vector)
        self.neighbor_vectors_id.append(neighbor_vector_id)
        self.neighbor_count = self.neighbor_count + 1
    
    def delete_neighbor_vector(self, pid):
        for i in range(self.neighbor_count):
            if pid == self.neighbor_vectors_id[i]:
                del self.neighbor_vectors[i]
                del self.neighbor_vectors_id[i]
                self.score_vector = np.delete(self.score_vector, i)
                self.neighbor_count = self.neighbor_count - 1
                print('Deletion...')
                break

    # Utility function to convert query_vector to torch tensor
    def convert_query_to_torch(self):
        if self.query_vector is not None:
            self.query_vector = torch.tensor(self.query_vector, dtype=torch.float64, requires_grad=False)

    # Utility function to convert score_vector to torch tensor
    def convert_score_to_torch(self):
        if self.score_vector is not None:
            self.score_vector = torch.tensor(self.score_vector, dtype=torch.float64, requires_grad=False)

    # Utility function to convert neighbor_vectors to torch tensors
    def convert_neighbors_to_torch(self):
        self.neighbor_vectors = [torch.tensor(neighbor, dtype=torch.float64, requires_grad=False) 
                                 for neighbor in self.neighbor_vectors]

    # Utility function to convert query_vector back to numpy array
    def convert_query_to_numpy(self):
        if isinstance(self.query_vector, torch.Tensor):
            self.query_vector = self.query_vector.numpy()

    # Utility function to convert score_vector back to numpy array
    def convert_score_to_numpy(self):
        if isinstance(self.score_vector, torch.Tensor):
            self.score_vector = self.score_vector.numpy()

    # Utility function to convert neighbor_vectors back to numpy arrays
    def convert_neighbors_to_numpy(self):
        if all(isinstance(neighbor, torch.Tensor) for neighbor in self.neighbor_vectors):
            self.neighbor_vectors = [neighbor.numpy() for neighbor in self.neighbor_vectors]

def get_catalogues_ids_per_doctor(catalogue_csv, doctor_csv, type, doctor_code):
    # Check if catalogues are the csv or just the filename
    if isinstance(catalogue_csv, str):
        catalogue_csv = pd.read_csv(catalogue_csv)
    if isinstance(doctor_csv, str):
        doctor_csv = pd.read_csv(doctor_csv)

    # Filter DataFrame for the specific doctor
    doctor_objects = doctor_csv[doctor_csv['User'] == doctor_code]
    catalogues = []

    # Iterate through every row in the doctor_objects DataFrame
    for index, row in doctor_objects.iterrows():
        catalogues.append(row['Catalogue ID'])
    
    # Initialize lists to hold query and retrieval data
    query = []
    retr = []

    # Determine the type of catalogue
    if type in ['E', 'G']:
        type_str = 'Ordered Exc Good Catalogue'
    elif type in ['P', 'F']:
        type_str = 'Ordered Fair Poor Catalogue'
    else:
        raise ValueError("Invalid type. Use 'E', 'G' for Excellent/Good and 'P', 'F' for Poor/Fair catalogues.")

    # Iterate through catalogues
    for i in range(len(catalogues)):
        row = catalogue_csv.loc[catalogue_csv['Catalogue ID'] == catalogues[i]]
        if pd.notna(row['Query Patient']).any() and pd.notna(row[type_str]).any():
            result_array = np.fromstring(row[type_str].values[0], sep=',')
            result_array = result_array.astype(int)
            if len(result_array) >= 2:
                query.append(row['Query Patient'].values[0])  # Adding the query patient value
                retr.append(result_array)  # Appending the result array as is

    # Return query and retr without converting retr to a NumPy array
    return query, retr

def get_pre_image_from_id(patient_info,favorite_images,patient_images,id):
  #Check if Catalogues are the csv or just the filename
  if isinstance(patient_info, str):
    patient_info = pd.read_csv(patient_info)
  if isinstance(favorite_images, str):
    favorite_images = pd.read_csv(favorite_images)
  if isinstance(patient_images, str):
    patient_images = pd.read_csv(patient_images)

  # Filter DataFrame for the specific doctor
  this_patient_info = patient_info[patient_info['Patient ID'] == id]
  this_favorite_images = favorite_images[favorite_images['Patient ID'] == id]
  pre_img=' '
  for index, row in this_patient_info.iterrows():
    surg_date=pd.to_datetime(row['Surgery Date'])

  for index, row in this_favorite_images.iterrows():
    date_difference = pd.to_datetime(row['Date'])-surg_date
    if((date_difference)<=pd.Timedelta(0)):
      pre_img=row['Image ID']

  this_patient_image = patient_images[patient_images['Image ID'] == pre_img]
  if(this_patient_image.empty):
    print('Patient:',this_patient_image)
    return None
  else:
    return this_patient_image['Image Filename'].item()

def get_tabular_data_from_id (patient_info, id):
  #Check if Catalogues are the csv or just the filename
  if isinstance(patient_info, str):
    patient_info = pd.read_csv(patient_info)
  tab_data=patient_info[patient_info['Patient ID'] == id]
  return tab_data

def get_tabular_data_from_id_numpy (patient_info, id):
  # Path to the uploaded CSV file
    data_as_strings = np.genfromtxt(patient_info, delimiter=',', dtype=str, encoding='utf-8', skip_header=1)
    # Ensuring data is 2D
    if data_as_strings.ndim == 1:
        data_as_strings = np.array([data_as_strings])
    # Convert id_to_match to string for comparison
    id_to_match_str = str(id)
    # Selecting rows where the value in the specified column matches the given ID (as string)
    selected_rows_as_strings = [row for row in data_as_strings if row[2] == id_to_match_str]
    # Display the selected rows
    return selected_rows_as_strings

def get_tabular_features_filtered_numpy(patient_info_csv, id, selected_features= [4, 5, 1, 6, 7]):

    full_info = get_tabular_data_from_id_numpy (patient_info_csv, id)
    selected_info = np.array([full_info[0][index] for index in selected_features])
    if '' in selected_info:
        return None
    height = np.float64(selected_info[0])
    weight = np.float64(selected_info[1])
    age    = convert_birthday_to_age    (selected_info[2])
    bs     = convert_brasize_to_underbust(selected_info[3])
    bc     = convert_bracup_to_oubustdiff(selected_info[4])
    if height is None or weight is None or age is None or bs is None or bc is None:
        return None

    selected_info[0] = height    
    selected_info[1] = weight    
    selected_info[2] = np.float64(age)
    selected_info[3] = np.float64(bs)
    selected_info[4] = np.float64(bc)

    selected_info = selected_info.astype(np.float64) 
    if np.any(selected_info == None):
        return None
    
    return selected_info

def get_tabular_features_filtered(patient_info_csv, id, selected_features= ['Patient Height', 'Patient Weight', 'Patient Birthday', 'Bra Size', 'Bra Cup']):

    full_info = get_tabular_data_from_id(patient_info_csv, id)
    selected_info = full_info[selected_features]
    
    if selected_info.isna().any().any():
        return None

    if 'Patient Birthday' in selected_features:
        selected_info.loc[:, 'Patient Birthday']  = selected_info['Patient Birthday'].apply(convert_birthday_to_age)

    if 'Bra Size' in selected_features:
        selected_info.loc[:, 'Bra Size']  = selected_info['Bra Size'].apply(convert_brasize_to_underbust)

    if 'Bra Cup' in selected_features:
        selected_info.loc[:, 'Bra Cup']  = selected_info['Bra Cup'].apply(convert_bracup_to_oubustdiff)
    
    if selected_info.isna().any().any():
        return None
    
    return selected_info.to_numpy()[0].astype(np.float64)

def get_catalogues_ids_from_csv(catalogue_csv,type):
  #Check if Catalogues are the csv or just the filename
  if isinstance(catalogue_csv, str):
    catalogue_csv = pd.read_csv(catalogue_csv)

  # Get the values of the 'Age' column
  Q_IDs = catalogue_csv['Query Patient']
  if(type=='E' or type =='G'):
      R_IDs = catalogue_csv['Ordered Exc Good Catalogue']
  if(type=='P' or type=='F'):
      R_IDs = catalogue_csv['Ordered Fair Poor Catalogue']
  query=[]
  retr=[]
  for i in range(len(Q_IDs)):
    # Check for NaN values before using np.fromstring
    if pd.notna(R_IDs.iloc[i]) and pd.notna(Q_IDs.iloc[i]):
      result_array = np.fromstring(R_IDs[i], sep=',')
      result_array = result_array.astype(int)
      if(len(result_array)>=2):
        retr.append(result_array)
        query.append(Q_IDs[i].item())
    # else:
    #   print('Null array in list!')
  #query = np.array(query)
  #retr = np.array(retr)
  return query, retr

def get_query_neighbor_elements(catalogue_info_csv, catalogue_user_info_csv, patient_info_csv, catalogue_type='E', doctor_code=-1):

    if doctor_code == -1:
        queries_id, neighbours_id = get_catalogues_ids_from_csv(catalogue_info_csv, catalogue_type)
    else:
        queries_id, neighbours_id = get_catalogues_ids_per_doctor(catalogue_info_csv, catalogue_user_info_csv, catalogue_type, doctor_code)
    
    # Remove repatative indexes
    for idx, q in enumerate(queries_id):
        for jdx, n in enumerate(neighbours_id[idx]):
            if q == n:
                neighbours_id[idx] = np.delete(neighbours_id[idx], jdx) #neighbours_id[idx].remove(n)

    QNS_list = []
    for idx in range(len(queries_id)): 
        qns_element = QNS_structure()
        #itm = get_tabular_features_filtered(patient_info_csv, queries_id[idx]) 
        itm = get_tabular_features_filtered_numpy(patient_info_csv, queries_id[idx]) 
        if itm is None:
            continue
        qns_element.set_query_vector(itm, queries_id[idx])
        
        for jdx in range(len(neighbours_id[idx])): 
            #itm = get_tabular_features_filtered(patient_info_csv, neighbours_id[idx][jdx]) 
            itm = get_tabular_features_filtered_numpy(patient_info_csv, neighbours_id[idx][jdx]) 
            if itm is None:
                continue
            qns_element.add_neighbor_vector(itm, neighbours_id[idx][jdx])
        qns_element.calculate_expert_score()
        QNS_list.append(qns_element)
        
    return QNS_list, len(QNS_list)

def get_query_neighbor_elements_path(catalogue_info_csv, catalogue_user_info_csv, patient_info_csv, favorite_image_info_csv, patient_images_info_csv, catalogue_type='E', doctor_code=-1):

    # Selecting Samples Based on DOCTOR Preference
    if doctor_code == -1:
        queries_id, neighbours_id = get_catalogues_ids_from_csv(catalogue_info_csv, catalogue_type)
    else:
        queries_id, neighbours_id = get_catalogues_ids_per_doctor(catalogue_info_csv, catalogue_user_info_csv, catalogue_type, doctor_code)
    
    # Remove repatative indexes
    for idx, q in enumerate(queries_id):
        for jdx, n in enumerate(neighbours_id[idx]):
            if q == n:
                neighbours_id[idx] = np.delete(neighbours_id[idx], jdx) #neighbours_id[idx].remove(n)

    QNS_list = []
    for idx in range(len(queries_id)): 
        qns_element = QNS_structure()
        itm = get_pre_image_from_id(patient_info_csv, favorite_image_info_csv, patient_images_info_csv,queries_id[idx])

        if itm is None:
            continue
        qns_element.set_query_vector(itm, queries_id[idx])
        
        for jdx in range(len(neighbours_id[idx])): 
            itm = get_pre_image_from_id(patient_info_csv, favorite_image_info_csv, patient_images_info_csv,neighbours_id[idx][jdx]) 
            if itm is None:
                continue
            qns_element.add_neighbor_vector(itm, neighbours_id[idx][jdx])
        qns_element.calculate_expert_score()
        QNS_list.append(qns_element)
        
    return QNS_list, len(QNS_list)

# Edit Resized Sample Names For the Algorithm
def edit_name_incase_using_resized(path, filename):
       basename = os.path.basename(filename)
       resized_filename = os.path.splitext(basename)[0] + '_resized.jpg'
       image_path = os.path.join(path, resized_filename)
       return image_path

def Sample_Manager(samples_path, train_pickle_path, test_pickle_path, catalogue_info, catalogue_user_info, patient_info, favorite_image_info, patient_images_info, catalogue_type='E', doctor_code=-1, split_ratio=0.8, default=True):

    if default:
        print('Reading Samples...')
        QNS_list, N = get_query_neighbor_elements_path(catalogue_info, catalogue_user_info, patient_info, favorite_image_info, patient_images_info,catalogue_type=catalogue_type, doctor_code=doctor_code) # 39 57 36 -1

        # In-case print to check
        #for q in QNS_list:
        #     q.show_summary()
        
        print('Shuffling Samples...')
        np.random.shuffle(QNS_list)

        print('Modifying File Addressing')
        for QNS_element in QNS_list:
            QNS_element.query_vector = edit_name_incase_using_resized(samples_path, QNS_element.query_vector)
            for j in range(0, QNS_element.neighbor_count):
                QNS_element.neighbor_vectors[j] = edit_name_incase_using_resized(samples_path, QNS_element.neighbor_vectors[j])

        print('Train-Test Split...')
        QNS_list_train = QNS_list[: int(np.floor(split_ratio * N))]
        QNS_list_test  = QNS_list[int(np.floor(split_ratio * N)):]

        print('Saving QNS...')
        with open(train_pickle_path, 'wb') as file:
            pickle.dump(QNS_list_train, file, protocol=pickle.DEFAULT_PROTOCOL)
        with open(test_pickle_path, 'wb') as file:
            pickle.dump(QNS_list_test, file, protocol=pickle.DEFAULT_PROTOCOL)
    else:   
        print('Loading QNS...')
        # Load the KMeans model from the file
        with open(train_pickle_path, 'rb') as file:
            QNS_list_train = pickle.load(file)
        with open(test_pickle_path, 'rb') as file:
            QNS_list_test = pickle.load(file)
    print('Done!')

    return QNS_list_train, QNS_list_test

#Required Paths
current_directory = os.getcwd()
images_path='../data/images/'
csvs_path ='../data/csvs/'
favorite_image_info = csvs_path + 'favorite_image_info.csv'
patient_info = csvs_path + 'patient_info.csv'
patient_images_info = csvs_path + 'patient_images.csv'
catalogue_info = csvs_path + 'catalogue_info.csv'
catalogue_user_info = csvs_path + 'catalogue_user_info.csv'
train_pickle_path = current_directory + '/../data/pickles/qns_list_train_F.pkl'
test_pickle_path  = current_directory + '/../data/pickles/qns_list_test_F.pkl'
path_save = '../bin/'
model_name = 'testings'

# Configs
np.random.seed(10)
torch.manual_seed(10)
device = "cuda:0" # "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
lr=0.00001
num_epochs=1
batch_size=16
margin = 0.0001
split_ratio=0.8
catalogue_type = 'E'
doctor_code=-1 # 39 57 36 -1

# Preprocessing
QNS_list_train, QNS_list_test = Sample_Manager(images_path, train_pickle_path, test_pickle_path, catalogue_info, catalogue_user_info, 
patient_info, favorite_image_info, patient_images_info, catalogue_type=catalogue_type, doctor_code=doctor_code, split_ratio=split_ratio, default=False)

# Down-Sampeling 
QNS_list_train = QNS_list_train[0:2]
QNS_list_test = QNS_list_test[0:2]

# Define Model
# model = Google_Base_Patch16_224()
# model = DeiT_Base_Patch16_224()
# model = Beit_Base_Patch16_224()
# model = DinoV2_Base_Patch16_224()


# model = ConvNextV2_Base_Patch16_224()
# model = NoPoolModifiedViT()
# model = Google_Base_Patch16_224_MLP()
# model  = ConvNext_Base() # WIERD ACCURACY!!!!
# model  = Swin_Base_Patch4_Window7_224()

# Define Dataset & Dataloaders & Optimization Parameters
train_dataset = TripletDataset(images_path, QNS_list_train, transform=model.get_transform())
test_dataset  = TripletDataset(images_path, QNS_list_test,  transform=model.get_transform())
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # later it should bea turned on ...
test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
criterion     = TripletMarginLoss(margin=margin, p=2)
optimizer     = optim.Adam(model.parameters(), lr=lr)

print('Training...')
model, _, _ = train_triplets(model, train_loader, test_loader, optimizer, criterion, num_epochs=num_epochs, device=device)

print('Evaluation...')
print(f'Train-NDDG: {evaluate_nddg(QNS_list_train, model, transform=model.get_transform(), device=device)[0]} Test-NDDG: {evaluate_nddg(QNS_list_test, model, transform=model.get_transform(), device=device)[0]}')

print('Saving...')
save_model(model, f'{path_save+model_name}.pl')
print('Done!')
