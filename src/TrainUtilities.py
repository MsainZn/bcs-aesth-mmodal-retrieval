import torch
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
import sys
import os
from itertools import combinations

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Creating The Triplets FOr Pytorch
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

# Apply The Training Session
def train_triplets(model, train_loader, test_loader, QNS_list_train, QNS_list_test, optimizer, criterion, num_epochs, model_name, device='cpu', path_save='../bin/'):
    # Open a log file in append mode
    os.makedirs(f'{path_save}{model_name}', exist_ok=True)
    with open(f'{path_save}{model_name}/Train_info.log', 'a') as f:
        # Redirect print statements to the file
        original_stdout = sys.stdout  # Save the original stdout
        sys.stdout = f
        print(f'{model}')
        model.to(device)
        model.train()  # Set model to training mode
        best_acc = float('-inf')  
        for epoch in range(num_epochs):
            running_loss = 0.0
            for data in train_loader:
                optimizer.zero_grad()  # Zero the parameter gradients
                queries = data['query'].to(device)
                positives = data['pos'].to(device)
                negatives = data['neg'].to(device)
                
                # Forward pass to get outputs and calculate loss
                anchor_embeddings = model(queries)
                pos_embeddings = model(positives)
                neg_embeddings = model(negatives)
                loss = criterion(anchor_embeddings, pos_embeddings, neg_embeddings)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * queries.size(0)
            
            # Evaluation
            epoch_loss = running_loss
            train_acc = evaluate_triplets(model, train_loader, device)
            test_acc  = evaluate_triplets(model, test_loader, device)

            current_time = datetime.now()
            print(f'[{current_time.strftime("%Y-%m-%d %H:%M:%S")}] Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train-Acc: {train_acc:.5f}, Test-Acc: {test_acc:.5f} Train-NDDG: {evaluate_nddg(QNS_list_train, model, transform=model.get_transform(), device=device)[0]} Test-NDDG: {evaluate_nddg(QNS_list_test, model, transform=model.get_transform(), device=device)[0]}')
    
            # Saving
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'{path_save}{model_name}/Epoch-{epoch}-{current_time.strftime("%Y-%m-%d %H_%M_%S")}-{best_acc:.5f}.pl')
                print(f"New best model saved with accuracy: {best_acc:.5f}")

        print('Finished Training')

        # Reset stdout to original
        sys.stdout = original_stdout

    return model, epoch_loss, epoch

# Evaluating Our Model using Triplet Loss
def evaluate_triplets(model, data_loader, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    total_triplets = 0
    correct_predictions = 0
    # total_pos_distance = 0.0
    # total_neg_distance = 0.0
    
    with torch.no_grad():  # No gradients needed
        for batch_idx, data in enumerate(data_loader):

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
            # total_pos_distance = total_pos_distance + pos_distances.sum().item()
            # total_neg_distance = total_neg_distance + neg_distances.sum().item()
            
            # Count correct predictions (positive distance should be less than negative distance)
            correct_predictions = correct_predictions + (pos_distances < neg_distances).sum().item()
            total_triplets = total_triplets + queries.size(0) # queries.size(0) len(queries)

            # print(f'Batch {batch_idx}:')
            # print(f'pos_distances: {pos_distances}')
            # print(f'neg_distances: {neg_distances}')
            # print(f'batch_correct_predictions: {(pos_distances < neg_distances).sum().item()}')
            # print(f'batch_triplet_count: {len(queries)}')
            # print(f'correct_predictions so far: {correct_predictions}')
            # print(f'total_triplets so far: {total_triplets}')
            # print('---')

    # Calculate average distances
    #avg_pos_distance = total_pos_distance / total_triplets
    #avg_neg_distance = total_neg_distance / total_triplets
    
    # Calculate accuracy
    accuracy = correct_predictions / total_triplets

    return accuracy    

# Evaluating Our Model Using nDDG Metric
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

# Function To calculate DDG Using Sorted Distances
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

# Save The Trained Model
def save_model(model, filepath):
    """
    Save the model's state dictionary to a file.

    Args:
    - model: The PyTorch model to save.
    - filepath: The path where the model will be saved.
    """
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

# Load A Trained Model
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

# def preprocess_single_sample(image_path, transform):
#     x = Image.open(image_path).convert('RGB')
#     x = transform(x)['pixel_values'].unsqueeze(0)
#     return x