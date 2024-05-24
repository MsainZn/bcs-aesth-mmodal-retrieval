
import torch
import numpy as np
from scipy.spatial.distance import euclidean
import os
import torch.nn as nn
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def euclidean_numpy(v1, v2):
    diff = v1 - v2 + 1e-15
    return np.sqrt(np.dot(diff,diff))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return x

# Define the Euclidean function using NumPy
def weighted_euclidean_numpy(v1, v2, W_mat, squared=True, abs_diff=True):
    
    diff_vec = v1 - v2 + 1e-15

    if abs_diff==True:
        diff_vec = np.abs(diff_vec)
    
    if squared == True:
        W_mat = W_mat ** 2
        
    return np.sqrt(np.dot(np.matmul(np.transpose(diff_vec), W_mat), diff_vec))

def dist_calc_selector_numpy (v1, v2, W_mat, dist_func=weighted_euclidean_numpy, is_mat=False, dim=0):

    if is_mat == False and dim == 0:
        W = np.diag(W_mat) + 1e-15
    elif dim != 0:
        W = np.matmul(np.transpose(W_mat), W_mat)
    else: 
        W = W_mat

    return dist_func(v1, v2, W)

def weighted_euclidean_torch(v1, v2, W_mat, squared=True, abs_diff=True):
    """
    Calculate the Euclidean distance between two vectors using PyTorch.
    Args:
    v1, v2 (torch.Tensor): Input tensors.

    Returns:
    torch.Tensor: The Euclidean distance.
    """
    diff_vec  = torch.sub(v1, v2) # + 1e-15 
    diff_vecM = diff_vec.view(diff_vec.shape[0], 1)
    diff_vecT = diff_vec.view(1, diff_vec.shape[0])

    if abs_diff==True:
        diff_vecM = np.abs(diff_vecM)
        diff_vecT = np.abs(diff_vecT)

    if squared == True:
        W_mat = W_mat ** 2

    return torch.sqrt(torch.matmul(torch.matmul(diff_vecT, W_mat), diff_vecM))

def dist_calc_selector_torch (v1, v2, W_mat, dist_func=weighted_euclidean_torch, is_mat=False, dim=0):

    if is_mat == False and dim == 0:
        W = torch.diag(W_mat) + 1e-15
    elif dim != 0:
        W = torch.matmul(torch.transpose(W_mat, 0, 1), W_mat)
    else: 
        W = W_mat

    return dist_func(v1, v2, W)

def sort_indexes_by_values(arr):
    """
    Sorts the indexes of a numpy array based on its values.

    Args:
    arr (np.array): The numpy array whose indexes are to be sorted.

    Returns:
    np.array: An array of indexes, sorted based on the values of the input array.
    """
    # Use np.argsort to get the sorted indices
    sorted_indices = np.argsort(arr)
    return sorted_indices

def Euclidian_optimizer(qns, distance_func=weighted_euclidean_torch, is_mat=False, dim=0, iweights=None, lr=0.001, num_epochs=3000, log_interval=20, loss_threshold=1e-15, patience=5):
    
    num_features = len(qns[0].query_vector)
    margin = 0.00001
    for q_element in qns:
        q_element.convert_query_to_torch()
        q_element.convert_neighbors_to_torch()
        q_element.convert_score_to_torch()

    if is_mat==True:
        weights = torch.eye(num_features, dtype=torch.float64) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)
    else:    
        weights = torch.ones(num_features, dtype=torch.float64) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)

    if iweights is not None:
        weights = torch.from_numpy(iweights)
        weights = weights.type(torch.float64).requires_grad_(True)
    
    if dim != 0:
        weights = torch.rand(dim, num_features) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)

    print("Initial weights:\n", weights)
    optimizer = torch.optim.Adam([weights], lr=lr)

    loss_values = []
    epochs_values = []
    patience_counter = 0
    
    epoch = 1
    loss = torch.zeros(1)
    while epoch < num_epochs:# or loss.item() > loss_threshold:
        optimizer.zero_grad()

        loss = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        for q_element in qns:
            for i in range(q_element.neighbor_count):
                for j in range(i + 1, q_element.neighbor_count):
                    dist_i = dist_calc_selector_torch(q_element.query_vector, q_element.neighbor_vectors[i], weights, distance_func, is_mat, dim)
                    dist_j = dist_calc_selector_torch(q_element.query_vector, q_element.neighbor_vectors[j], weights, distance_func, is_mat, dim)
                    cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                    # print(f'i = {i:03}, j = {j:03}, dist-qi = {dist_i.item():010.07f}, dist-qj = {dist_j.item():010.07f}, Score-i: {q_element.score_vector[i].item():02.0f}, Score-j: {q_element.score_vector[j].item():02.0f} Update-Loss: {cond.item()}')
                    if cond:
                        loss = loss + torch.abs(dist_i - dist_j) + 1.00 / (torch.sum(torch.abs(weights)) + margin)
                        # loss = loss + F.relu(margin + (dist_i - dist_j)) / (torch.sum(weights) + margin)
        
        loss_values.append(loss.item())
        epochs_values.append(epoch)
        
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch} loss is: {loss.item():.15f}')
           # print(weights)

        if epoch > 1 and abs(loss_values[-1] - loss_values[-2]) < loss_threshold:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Loss converged {loss_values[-1]:.12} in {epochs_values[-1]} epochs.")
                break
        else:
            patience_counter = 0

        loss.backward()
        optimizer.step()
        epoch = epoch + 1

    return weights.detach().numpy(), loss, epochs_values

def evaluate_Euclidian(QNS_list, optimizsed_weights, distance_func_numpy=weighted_euclidean_numpy, is_mat=False, dim=0):
    first_order  = []
    final_order  = []
    target_order = []

    for qns in QNS_list:
        fw = []
        for neighbor in qns.neighbor_vectors:
            fw.append(dist_calc_selector_numpy(qns.query_vector, neighbor, optimized_weights, dist_func=distance_func_numpy, is_mat=is_mat, dim=dim))
            
        first_order.append(qns.calculate_initial_sort().tolist())
        final_order.append(sort_indexes_by_values(fw).tolist())
        target_order.append(qns.calculate_expert_sort().tolist())

    # for idx in range(len(final_order)):
    #     print(f'Distance-based: {first_order[idx]}')
    #     print(f'Target Order:   {target_order[idx]}')
    #     print(f'Final Order:    {final_order[idx]}')
    #     print('****************************')

    model_acc = 100 * np.mean(test_ndcg(final_order))
    base_acc  = 100 * np.mean(test_ndcg(first_order))

    return model_acc, base_acc, first_order, final_order, target_order

def MLP_optimizer(qns, hidden_dim=2, output_dim=16, lr=0.01, num_epochs=200, log_interval=1, loss_threshold=1e-15, patience=5):

    best_accuracy = 0.0  # Initialize best accuracy
    num_features = len(qns[0].query_vector)
    margin = 0.0001    
    mlp_model = MLP(num_features, hidden_dim, output_dim).double() 
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(mlp_model.parameters(), lr=lr, momentum=0.9)
    loss_values = []
    patience_counter = 0
    
    epoch = 1
    loss = torch.zeros(1)
    while epoch < num_epochs:
        optimizer.zero_grad()
        loss = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        
        for q_element in qns:
            # Query Vec Prep
            query_vec = torch.tensor(q_element.query_vector, dtype=torch.float64).unsqueeze(0)  # Add batch dimension
            query_vec_transformed = mlp_model(query_vec.unsqueeze(0))  # Transform query vector

            # Neighbour Vec Prep
            for i in range(q_element.neighbor_count):
                for j in range(i + 1, q_element.neighbor_count):
                    # Transform neighbor i vector & Use dist_calc_selector_torch with transformed vectors
                    neighbor_i = torch.tensor(q_element.neighbor_vectors[i], dtype=torch.float64).unsqueeze(0)  # Add batch dimension
                    neighbor_i_transformed = mlp_model(neighbor_i.unsqueeze(0)) 
                    score_i = torch.tensor(q_element.score_vector[i], dtype=torch.float64).unsqueeze(0)
                    dist_i = torch.norm(query_vec_transformed - neighbor_i_transformed)

                    # Transform neighbor j vector & Use dist_calc_selector_torch with transformed vectors
                    neighbor_j = torch.tensor(q_element.neighbor_vectors[j], dtype=torch.float64).unsqueeze(0)
                    neighbor_j_transformed = mlp_model(neighbor_j.unsqueeze(0))
                    score_j = torch.tensor(q_element.score_vector[j], dtype=torch.float64).unsqueeze(0)
                    dist_j = torch.norm(query_vec_transformed - neighbor_j_transformed)
                    cond = (score_i > score_j and dist_i < dist_j) or (score_i < score_j and dist_i > dist_j)
                    
                    #dist_i = dist_calc_selector_torch(query_vec_transformed, neighbor_i_transformed, torch.ones(output_dim, dtype=torch.float64), weighted_euclidean_torch, False, 0)
                    #dist_j = dist_calc_selector_torch(query_vec_transformed, neighbor_j_transformed, torch.ones(output_dim, dtype=torch.float64), weighted_euclidean_torch, False, 0)
                    
                    #print(f'i = {i:03}, j = {j:03},\n Neighbor-i = {neighbor_i}\n, Neighbor-j = {neighbor_j}')
                    #print(f'i = {i:03}, j = {j:03}\n \nQuery =      {query_vec_transformed}\nNeighbor-i = {neighbor_i_transformed}\nNeighbor-j = {neighbor_j_transformed}')

                    #print(f'i = {i:03}, j = {j:03}, Query-Neighbor(i) = {query_vec_transformed - neighbor_i_transformed}, Query-Neighbor(j) = {query_vec_transformed - neighbor_j_transformed}')
                    #print(f'i = {i:03}, j = {j:03}, dist-qi = {dist_i.item():010.07f}, dist-qj = {dist_j.item():010.07f}, Score-i: {score_i.item():02.0f}, Score-j: {score_j.item():02.0f} Update-Loss: {cond.item()}')

                    if cond:
                        loss = loss + torch.abs(dist_i - dist_j) #+ 1.00 / (torch.sum(torch.abs(mlp_model.fc2.weight)) + margin)

        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

        model_acc, base_acc, _, _, _ = evaluate_MLP(qns, mlp_model)
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch} Loss: {loss.item():.15f} Model: {model_acc:.5f} Base: {base_acc:.5f}')
           # print(weights)

        if model_acc > best_accuracy:
            best_accuracy = model_acc  # Update best accuracy
            best_mlp_model= mlp_model
            torch.save(best_mlp_model.state_dict(), 'best_model.pl')  # Save model state
            print(f"Epoch {epoch}: New best model saved with accuracy: {best_accuracy:.4f}")
        else:
            mlp_model = best_mlp_model

        epoch = epoch + 1

    # for param in mlp_model.parameters():
    #     print(param.data)

    return best_mlp_model, loss.item(), epoch

def evaluate_MLP(QNS_list, model):
    
    first_order_value  = []
    final_order_value  = []

    first_order  = []
    final_order  = []
    target_order = []

    model.eval()

    with torch.no_grad():  # No need to track gradients during evaluation
        for qns in QNS_list:
            # Convert query vector to tensor if it's not already
            if not isinstance(qns.query_vector, torch.Tensor):
                query_vector = torch.tensor(qns.query_vector, dtype=torch.float64).unsqueeze(0)  # Add batch dimension

            # Transform the query vector using the MLP model and convert to numpy array
            query_vec_transformed = model(query_vector).squeeze()

            fss = []
            iss = []
            for neighbor in qns.neighbor_vectors:
                # Convert neighbor vector to tensor if it's not already
                if not isinstance(neighbor, torch.Tensor):
                    neighbor_vector = torch.tensor(neighbor, dtype=torch.float64).unsqueeze(0)  # Add batch dimension
                # Transform the neighbor vector using the MLP model and convert to numpy array
                neighbor_transformed = model(neighbor_vector).squeeze()
                distf = torch.norm(query_vec_transformed - neighbor_transformed)
                fss.append(distf.numpy())  # Assuming dist_calc_selector_numpy returns a scalar distance value
                
                disti = torch.norm(query_vector - neighbor_vector)
                iss.append(disti.numpy())  # Assuming dist_calc_selector_numpy returns a scalar distance value
            
            # Use argsort to sort indices based on calculated distances
            first_order.append(np.argsort(iss).tolist()) 
            final_order.append(np.argsort(fss).tolist()) 
            target_order.append(qns.calculate_expert_sort().tolist())

            first_order_value.append(iss)
            final_order_value.append(fss) 

    # Calculate NDCG or other evaluation metrics based on first_order, final_order, and target_order
    # Assuming test_ndcg is defined elsewhere and calculates the NDCG based on the provided orders
    # for idx in range(len(final_order)):
    #     print(f'Distance-based: {first_order[idx]}')
    #     print(f'Target Order:   {target_order[idx]}')
    #     print(f'Final Order:    {final_order[idx]}')
    #     print('****************************')
    
    base_acc  = 100 * np.mean(test_ndcg(first_order))
    model_acc = 100 * np.mean(test_ndcg(final_order))
    
    return model_acc, base_acc, first_order_value, final_order_value, target_order

np.random.seed(10)
torch.manual_seed(10)

distance_func_torch = weighted_euclidean_torch # weighted_cosine_similarity_torch #
distance_func_numpy = weighted_euclidean_numpy # weighted_cosine_similarity_numpy #
is_mat=True
lr=0.001
num_epochs=100
log_interval=1
dim = 3
loss_threshold = 1e-12

print('Readin Samples From Dataset...')
QNS_list, N = get_query_neighbor_elements('csvs/catalogue_info.csv', 'csvs/catalogue_user_info.csv', 'csvs/patient_info.csv', doctor_code=-1) # 39 57 36 -1

#normalize_across_all_qns_feature_based(QNS_list)
# for q in QNS_list:
#     q.show_summary()

print('Shuffling Samples...')
np.random.shuffle(QNS_list)
QNS_list_train = QNS_list[: int(np.floor(0.8 * N))]
QNS_list_test  = QNS_list[int(np.floor(0.8 * N)):]

print('Training...')
model, loss, ep = MLP_optimizer(QNS_list_train, hidden_dim=16, output_dim=5, lr=lr, num_epochs=50, log_interval=1, loss_threshold=1e-15, patience=5)
model_acc, base_acc, first_order, final_order, target_order = evaluate_MLP(QNS_list_test, model)
print(f'Base Model Testset: {base_acc}')
print(f'MLP  Model Testset: {model_acc}')

print('Evaluating...')
for idx in range(len(final_order)):
    # print(f'Distance-based: {first_order[idx]}')
    # print(f'Target Order:   {target_order[idx]}')
    print(f'Final Order:    {final_order[idx]}')
    print('****************************')

# for param in model.parameters():
#     print(param.data)

#optimized_weights, final_loss, epochs_run = Euclidian_optimizer(QNS_list_train, distance_func_torch, is_mat=is_mat, dim=dim, lr=lr, num_epochs=200, loss_threshold=loss_threshold, log_interval=log_interval)
#evaluate_Euclidian(QNS_list_test, optimized_weights, is_mat=is_mat, dim=dim)