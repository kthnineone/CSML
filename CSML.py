# -*- coding: utf-8 -*-
"""
# Methods
CSML

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import random
import pickle
import time

# Default tensor type as Double, float64
torch.set_default_tensor_type(torch.DoubleTensor)


# Constrain the embeddings
def normalization(x):
    n_x = x.size()[0]
    n_dim = x.size()[1]
    ones_vector = torch.ones(n_x, device = device).double()
    norms_vector = torch.norm(x, p=2, dim=1).double()
    norms_vector = torch.max(ones_vector, norms_vector)
    norms_matrix = (norms_vector.repeat(n_dim, 1)).transpose(0,1)
    x = x / norms_matrix
    return x

# Triplet loss
def triplet_loss(margin, anchor, positive, negative):
    M = anchor.shape[0]
    N = positive.shape[0]
    U = negative.shape[0]
    embed_dim = anchor.shape[1]
    # location of embed_dim
    d_loc = len(anchor.shape)-1
    if M != 1:
        anchor = torch.unsqueeze(anchor, dim = 1)
        d_loc = len(anchor.shape)-1
    # Distance of anchor and positive    
    dist1 = (anchor - positive)**2 
    dist1 = dist1.sum(dim = d_loc) 
    d_loc = len(dist1.shape)
    # Distance of anchor and negative
    dist2 = (anchor - negative)**2 
    dist2 = dist2.sum(dim = d_loc) 

    dist1 = torch.unsqueeze(dist1, dim = d_loc)
    # Hinge loss 
    hinge_loss = margin + dist1 - dist2
    zero = torch.DoubleTensor([0]).to(device)
    # Triplet loss
    triplet_margin_loss = torch.max(zero, hinge_loss)
    triplet_margin_loss = triplet_margin_loss.sum()
    return triplet_margin_loss

# Update user and item embeddings
def change_parm(model):
    model = model
    model_normalize = normalization(model.weight)
    model_normalize = model_normalize.cpu()
    model_normalize = model_normalize.detach()
    model_normalize = model_normalize.numpy()
    model.weight.data.copy_(torch.from_numpy(model_normalize))
    
# Initiliaze margins
def init_margin(margin):
    n = margin.weight.size()[0]
    one = torch.FloatTensor([1.0])
    ones = one.repeat(n)
    ones = torch.unsqueeze(ones, 1)
    margin.weight.data.copy_(ones)

# Normalize margins
def norm_margin(margin):
    clipped = torch.clamp(margin.weight, 0, 1.0)
    margin.weight.data.copy_(clipped)

class CSML_class(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, 
                train_u_lists, neg_uv_lists, neg_lists,
                w1, w2, w3):
        super(CSML_class, self).__init__()
        self.train_uv_lists = train_u_lists
        self.neg_uv_lists = neg_uv_lists # for learning
        self.neg_lists = neg_lists # for testing
        #Initialize the embeddings
        self.u2e = nn.Embedding(num_users, embed_dim).to(device)
        self.v2e = nn.Embedding(num_items, embed_dim).to(device)
        # Initialize the values of embeddings from N[0.01, 0.03]
        self.u2e_init = torch.normal(0.1, 0.03, size = (num_users, embed_dim))
        self.u2e.weight.data.copy_(self.u2e_init)
        self.v2e_init = torch.normal(0.1, 0.03, size = (num_items, embed_dim))
        self.v2e.weight.data.copy_(self.v2e_init)
        # Initialize the margins
        self.margin_uv = nn.Embedding(num_users, 1).to(device)
        self.margin_vv  = nn.Embedding(num_items, 1).to(device)
        self.margin_uu  = nn.Embedding(num_users, 1).to(device)
        # initilization the margins
        # All the initial margin values are 1
        init_margin(self.margin_uv)
        init_margin(self.margin_vv)
        init_margin(self.margin_uu)
        # Hyperparameters for losses
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
    
    def forward(self, train_u):
        total_loss = 0
        batch_item_lists = set()
        for i, user in enumerate(train_u):
            user_value = user.item()
            if user_value in all_users:
                # make embeddings for user, pos and neg items
                user_embed = self.u2e.weight[user_value]
                user_embed = torch.unsqueeze(user_embed, 0)
                positive = self.train_uv_lists[user.item()]
                positive = torch.LongTensor(positive).to(device)
                pos_embed = self.v2e.weight[positive]
                negative = self.neg_uv_lists[user_value]
                negative = self.neg_uv_lists[user.item()]
                negative = torch.LongTensor(negative).to(device)
                neg_embed = self.v2e.weight[negative]
                # load the margin
                margin_u = self.margin_uv.weight[user_value]
                #user-centric loss
                uv_loss = triplet_loss(margin_u, user_embed, pos_embed, neg_embed)
                batch_item_lists = batch_item_lists.union(set(positive.tolist()))
                # load the margin for each item
                margin_v = self.margin_vv.weight[positive]
                #item-centric loss
                vv_loss = triplet_loss(margin_v, pos_embed, user_embed, neg_embed)
                total_loss += uv_loss + (self.w1*vv_loss)        
                # socio-centric loss
                if user.item() in nb_users:
                    neighbor = np.int64(social_adj_lists[user.item()])
                    neighbor = torch.LongTensor(neighbor).to(device)    
                    nb_embed = self.u2e.weight[neighbor]
                    non_neighbor = np.int64(neg_social_adj_lists[user.item()])
                    non_neighbor = torch.LongTensor(non_neighbor).to(device)
                    non_nb_embed = self.u2e.weight[non_neighbor]
                    # load the marings for each loss
                    margin_s = self.margin_uu.weight[user.item()]
                    uu_loss = triplet_loss(margin_s, user_embed, nb_embed, non_nb_embed)
                    total_loss += (self.w3*uu_loss)

        # calculate L_am
        batch_item_lists = list(batch_item_lists)
        loss_am = (self.margin_uv.weight[train_u]).mean()
        loss_am += (self.margin_vv.weight[batch_item_lists]).mean()
        loss_am += (self.margin_uu.weight[train_u]).mean()
        total_loss += (self.w2*loss_am)
        return total_loss

    

# Precision
# Recall
# MRR
# NDCG

def metrics(user, recommended_items, pos_items, k_list):
    precision = 0.0
    recall = 0.0
    MRR = 0.0
    DCG = 0.0
    
    larger3_n = len(test_u_lists[user.item()])
    n_p = len(pos_items)
    #print(n_p)
    topk_result = {}
    
    if n_p > 0 and larger3_n > 0:
        for k in k_list:
            correct_item_count = 0
            MRR = 0.0
            DCG = 0.0
            if n_p > k:
                n_idcg = k
            else:
                n_idcg = n_p
            # Create the value of IDCG
            IDCG = 0
            #print(n_idcg, k)
            for j in range(n_idcg):
                IDCG += 1/np.log2(j+2)
                
            topk = recommended_items[0:k]
            for i, item in enumerate(topk):
                if item.item() in pos_items:
                    correct_item_count += 1
                    MRR += 1/(i+1)
                    DCG += 1/np.log2(i+2)
            NDCG = DCG / IDCG
            MRR /= k
            precision = correct_item_count/k
            recall = correct_item_count/n_p
            recall = correct_item_count/larger3_n
            #k_dict = {k: [precision, recall, trc_recall, MRR, NDCG]}
            topk_result[k] = np.array([precision, recall, MRR, NDCG])        
    else:
        NDCG = 0.0
        for k in k_list:
            topk_result[k] = np.array([precision, recall, MRR, NDCG])  

    return topk_result



def recommend(test_users, neg_lists, k_list, u2e, v2e):
    
    P = 0.0
    R = 0.0
    MRR = 0.0
    NDCG = 0.0

    result = {}
    k_list = [5, 10, 20]
    for k in k_list:
        result[k] = np.array([P, R, MRR, NDCG])
    
    
    all_users = list(history_u_lists.keys())
    for i, user in enumerate(test_users):
        if user.item() in all_users:
            user_embed = u2e.weight[user.item()]
            # test items 
            positive = test_u_lists[user.item()]
            positive = torch.LongTensor(positive)
            pos_embed = v2e.weight[positive]
            # negative items
            negative = neg_lists[user.item()]
            negative = torch.LongTensor(negative)
            neg_embed = v2e.weight[negative]
            item_list = torch.cat([positive, negative])
            # calculate the distance of user and pos item
            dist1 = (user_embed - pos_embed)**2 
            dist1 = dist1.sum(dim=1)
            # calculate the distance of user and neg item
            dist2 = (user_embed - neg_embed)**2
            dist2 = dist2.sum(dim=1)
            # sort them in ascending order
            dist = torch.cat([dist1, dist2])
            sort, indices = torch.sort(dist)

            sorted_items = item_list[indices]
            
            k_list = [5, 10, 20]
            
            user_results = metrics(user, sorted_items, positive, k_list)
            for k in k_list:
                result[k] += user_results[k]
    # Divide by the size of users
    for k in k_list:
        result[k] /= len(test_users)
        
    for k in k_list:    
        print("for Top-{0}".format(k))
        pk = result[k][0]
        rk = result[k][1]
        mk = result[k][2]
        nk = result[k][3]
        print("Precision: {0}, Recall: {1}".format(round(pk, 4), round(rk, 4)))
        print("MRR: {0}, NDCG: {1}".format(round(mk, 4), round(nk, 4)))
    
    return nk


### Training

def training_function(
        train_uv_lists, neg_uv_lists, neg_lists
        ,w1_1, w2_1, w3_1,
        num_users1, num_items1, n_epochs1,
        learning_rate1, device1,
        test_data,
        minibatch_size):
    
    num_users = num_users1
    num_items = num_items1
    embed_dim = 100
    learning_rate = learning_rate1
    device = device1
    n_epochs = n_epochs1
    # Weights of losses   
    w1 = w1_1
    w2 = w2_1
    w3 = w3_1
    
    model = CSML_class(num_users, num_items, embed_dim, 
                 train_uv_lists, neg_uv_lists, neg_lists,
                w1, w2, w3)
    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate )

    optimizer.zero_grad()   
    min_loss = torch.FloatTensor([10**10+1.0])
    
    epoch_loss = 0
    endure_count = 0
    k_list = [5, 10, 20]
    
    loss_list = []
    
    for epoch in range(n_epochs):
        start = time.time()
        epoch_loss = 0
        # shuffle the train data per epoch
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True)
        for batch in train_loader:
            optimizer.zero_grad()
            loss = 0
            user_list = batch[0]
            loss = loss + model.forward(user_list)
            # Compute the triplet loss
            loss.backward(retain_graph=True)
            optimizer.step()
            with torch.no_grad():
                # normalization
                change_parm(model.u2e)
                change_parm(model.v2e)
                # pytorch version of tf.clip_by_value
                norm_margin(model.margin_uv)
                norm_margin(model.margin_vv)
                norm_margin(model.margin_uu)
            
            with torch.no_grad():
                epoch_loss += loss
        
        with torch.no_grad():
            temp_loss = epoch_loss.cpu().clone()
            loss_list.append([temp_loss])
            prev_min_loss = min_loss.clone()
            min_loss = min(temp_loss, min_loss)
                
        print("----------------------------------------")
        print("current epoch is ",epoch, "and loss is ", round(epoch_loss.item(), 4))
        print("current min loss is ", round(min_loss.item(), 4))
        with torch.no_grad():
            ndcg10 = recommend(test_data, neg_lists, k_list, model.u2e, model.v2e)
        end = time.time()
        duration = round((end - start)/60, 4)
        print("Taken time {} min ".format(duration))

        if prev_min_loss <= min_loss:
            endure_count += 1
            
    return loss_list

    
'''
Yelp Hotel Example
'''

path = ''


with open(path+'train_u_lists.pickle', 'rb') as fr:
    train_u_lists = pickle.load(fr)

with open(path+'test_u_lists.pickle', 'rb') as fr:
    test_u_lists = pickle.load(fr)
    
# Trust relation in dictionary type
with open(path+'social_adj_lists.pickle', 'rb') as fr:
    social_adj_lists = pickle.load(fr) 

# U = 499, unconnected users
with open(path+'neg_social_adj_lists.pickle', 'rb') as fr:
    neg_social_adj_lists = pickle.load(fr)     

# Negative items for testing
with open(path+'all_neg_lists.pickle', 'rb') as fr:
    all_neg_lists = pickle.load(fr) 
    
# Negative items for training
with open(path+'neg_uv_lists499.pickle', 'rb') as fr:
    neg_uv_lists499 = pickle.load(fr) 

    
w1_curr = 0.1
w2_curr = 0.1
w3_curr = 1
lr = 0.05

#device1 = "cuda"
device1 = "cpu"

# All users in train_u_listss
all_users = list(history_u_lists.keys())
# All trustors in social_adj_lists
nb_users = list(social_adj_lists.keys())

arr = np.array(list(history_u_lists.keys()))
data = torch.LongTensor(arr)
train_u = torch.LongTensor(arr).to(device)
val_u = torch.LongTensor(arr).to(device)
test_u = torch.LongTensor(arr).to(device)
    
trainset = torch.utils.data.TensorDataset(train_u)
valset = torch.utils.data.TensorDataset(val_u)
testset = torch.utils.data.TensorDataset(test_u)

n_users = 1100
n_items = 1487

num_epochs = 100
minibatch_size = 128

training_function(
            train_u_lists, neg_uv_lists499, all_neg_lists,
            w1_curr, w2_curr, w3_curr,
            n_users, n_items, num_epochs,
            lr, device1,
            all_users,
            minibatch_size)
    