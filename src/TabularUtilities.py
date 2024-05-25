import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import BatchNorm1d
import numpy
from TrainUtilities import test_ndcg 
import copy
from scipy.spatial.distance import euclidean


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.bn2 = BatchNorm1d(2*hidden_dim) 
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim) 
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.sigmoid(self.bn1(self.fc1(x)))
        x = F.sigmoid(self.bn2(self.fc2(x)))
        x = F.sigmoid(self.bn3(self.fc3(x)))
        x = F.sigmoid(self.fc4(x))
        return x
    
    def get_transform(self):
        def transform(tabular_vector):
            x = torch.tensor(tabular_vector, dtype=torch.float32).squeeze(0)
            return x
        return transform

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
        diff_vecM = numpy.abs(diff_vecM)
        diff_vecT = numpy.abs(diff_vecT)

    if squared == True:
        W_mat = W_mat ** 2

    return torch.sqrt(torch.matmul(torch.matmul(diff_vecT, W_mat), diff_vecM))

def eucl_model_manager_torch (v1, v2, W_mat, is_mat=False, dim=0):

    if is_mat == False and dim == 0:
        W = torch.diag(W_mat) + 1e-15
    elif dim != 0:
        W = torch.matmul(torch.transpose(W_mat, 0, 1), W_mat)
    else: 
        W = W_mat

    return weighted_euclidean_torch(v1, v2, W)

def euclidian_optimizer(QNS_list_train, QNS_list_test, is_mat=False, dim=0, lr=0.0001, num_epochs=100, margin = 0.00001):

    qns = copy.deepcopy(QNS_list_train)
    num_features = len(qns[0].query_vector)
    for q_element in qns:
        q_element.convert_query_to_torch()
        q_element.convert_neighbors_to_torch()
        q_element.convert_score_to_torch()
    
    # Configuring Tthe weights based on the matrix format
    if is_mat==True:
        # weights = torch.eye(num_features, dtype=torch.float64) + 1e-15
        weights = torch.rand(num_features, num_features) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)
    else:    
        # weights = torch.ones(num_features, dtype=torch.float64) + 1e-15
        weights = torch.rand(num_features, dtype=torch.float64) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)
    
    if dim != 0:
        weights = torch.rand(dim, num_features) + 1e-15
        weights = weights.type(torch.float64).requires_grad_(True)

    # print("Initial weights:\n", weights)
    optimizer = torch.optim.Adam([weights], lr=lr)

    final_ordering = []
    for epoch in range(0, num_epochs):
        optimizer.zero_grad()
        success_count = 0
        total_count = 0
        loss = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        for q_element in qns:
            qn_dist = []
            for i in range(q_element.neighbor_count):
                for j in range(i + 1, q_element.neighbor_count):
                    dist_i = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[i], weights, is_mat, dim)
                    dist_j = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[j], weights, is_mat, dim)
                    
                    cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                    # print(f'i = {i:03}, j = {j:03}, dist-qi = {dist_i.item():010.07f}, dist-qj = {dist_j.item():010.07f}, Score-i: {q_element.score_vector[i].item():02.0f}, Score-j: {q_element.score_vector[j].item():02.0f} Update-Loss: {cond.item()}')
                    if cond:
                        loss = loss + torch.abs(dist_i - dist_j) + 1.00 / (torch.sum(torch.abs(weights)) + margin)
                        # loss = loss + F.relu(margin + (dist_i - dist_j)) / (torch.sum(weights) + margin)
                    else:
                        success_count = success_count + 1
                    total_count = total_count + 1
                qn_dist.append(dist_i.item())
            final_ordering.append(qn_dist)
            
        acc_train = success_count/total_count
        ddg_train = 100 * numpy.mean(test_ndcg(final_ordering))
        final_ordering.clear()

        copied_tensor = weights.detach().clone().numpy()
        acc_test, ddg_test = euclidian_evaluate(QNS_list_test, copied_tensor, is_mat=is_mat, dim=dim)
        
        print(f'Epoch {epoch} loss is: {loss.item():.10f} Train-Acc: {acc_train:.6} Train-DDG: {ddg_train:.6} Test-Acc: {acc_test:.6} Test-DDG: {ddg_test:.6}')

        loss.backward()
        optimizer.step()

    print('Summary:')
    copied_tensor = weights.detach().clone().numpy()
    acc_base_train, ddg_base_train = euclidian_base(QNS_list_train)
    acc_train, ddg_train           = euclidian_evaluate(QNS_list_train, copied_tensor, is_mat=is_mat, dim=dim)
    print(f'Trainset Raw-Euclidian Acc: {acc_base_train:.6} and DDG: {ddg_base_train:.6} | Model-Acc: {acc_train:.6} Model-DDG: {ddg_train:.6}!')
    print(f'Trainset Rate of Improvement: Acc: {100*(acc_train-acc_base_train)/acc_base_train:.6}% and DDG: {100*(ddg_train-ddg_base_train)/ddg_base_train:.6}%')
    
    acc_base_test, ddg_base_test = euclidian_base(QNS_list_test)
    acc_test, ddg_test           = euclidian_evaluate(QNS_list_test, copied_tensor, is_mat=is_mat, dim=dim)
    print(f'Testset Raw-Euclidian Acc: {acc_base_test:.6} and DDG: {ddg_base_test:.6} | Model-Acc: {acc_test:.6} Model-DDG: {ddg_test:.6}!')
    print(f'Testset Rate of Improvement: Acc: {100*(acc_test-acc_base_test)/acc_base_test:.6}% and DDG: {100*(ddg_test-ddg_base_test)/ddg_base_test:.6}%')
    
    return weights.detach().numpy()

def euclidian_evaluate(QNS_list, iweights, is_mat, dim=0):
    
    # acc_base, ddg_base = euclidian_base(QNS_list)
    # print(f'The Dataset Base Accuracy: {acc_base:.4} and DDG: {ddg_base:.4}!')

    qns = copy.deepcopy(QNS_list)
    # Preparing Testset into right format
    for q_element in qns:
        q_element.convert_query_to_torch()
        q_element.convert_neighbors_to_torch()
        q_element.convert_score_to_torch()
    
    # Preparing the weights into right format
    weights = torch.from_numpy(iweights)
    weights = weights.type(torch.float64)

    # Accuracy Evaluation
    final_ordering = []
    success_count = 0
    total_count = 0

    # Evaluation Loop
    for q_element in qns:
        qn_dist = []
        for i in range(q_element.neighbor_count):
            dist_i = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[i], weights, is_mat, dim)
            for j in range(i + 1, q_element.neighbor_count):
                dist_j = eucl_model_manager_torch(q_element.query_vector, q_element.neighbor_vectors[j], weights, is_mat, dim)
                cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                # print(f'i = {i:03}, j = {j:03}, dist-qi = {dist_i.item():010.07f}, dist-qj = {dist_j.item():010.07f}, Score-i: {q_element.score_vector[i].item():02.0f}, Score-j: {q_element.score_vector[j].item():02.0f} Update-Loss: {cond.item()}')
                if cond == False:
                    success_count = success_count + 1   
                total_count = total_count + 1
            qn_dist.append(dist_i.item())
        final_ordering.append(qn_dist)
    
    acc = success_count/total_count
    ddg = 100 * numpy.mean(test_ndcg(final_ordering))
    # print(f'Acc-Current: {acc:.4} ({100*(acc-acc_base)/acc_base:.4}%) | DDG-Current: {ddg:.4} ({100*(ddg-ddg_base)/ddg_base:.4}%)')
    return acc, ddg

def euclidian_base(qns):
    # Accuracy Evaluation
    final_ordering = []
    success_count = 0
    total_count = 0

    # Evaluation Loop
    for q_element in qns:
        qn_dist = []
        for i in range(q_element.neighbor_count):
            dist_i = euclidean(q_element.query_vector, q_element.neighbor_vectors[i])
            for j in range(i + 1, q_element.neighbor_count):
                dist_j = euclidean(q_element.query_vector, q_element.neighbor_vectors[j])
                cond = (q_element.score_vector[i] > q_element.score_vector[j] and dist_i < dist_j) or (q_element.score_vector[i] < q_element.score_vector[j] and dist_i > dist_j)
                # print(f'i = {i:03}, j = {j:03}, dist-qi = {dist_i.item():010.07f}, dist-qj = {dist_j.item():010.07f}, Score-i: {q_element.score_vector[i].item():02.0f}, Score-j: {q_element.score_vector[j].item():02.0f} Update-Loss: {cond.item()}')
                if cond == False:
                    success_count = success_count + 1   
                total_count = total_count + 1
            qn_dist.append(dist_i)
        final_ordering.append(qn_dist)
    
    acc = success_count/total_count
    ddg = 100 * numpy.mean(test_ndcg(final_ordering))
    # print(f'Accuracy: {acc:.4}  DDG: {ddg:.4}')
    return acc, ddg