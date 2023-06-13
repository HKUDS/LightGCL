import numpy as np
import torch
import pickle
from model import LightGCL
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parser import args
from tqdm import tqdm
import time
from setproctitle import setproctitle
import os
os.environ["TORCH_AUTOGRAD_SHUTDOWN_WAIT_LIMIT"] = "0"

setproctitle('EXP@Xuheng')

device = 'cuda:' + args.cuda

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
svd_q = args.q

# load data
path = 'data/' + args.data + '/'
f = open(path+'trnMat.pkl','rb')
train = pickle.load(f)
#train_np = train.toarray()
train_csr = (train!=0).astype(np.float32)
f = open(path+'tstMat.pkl','rb')
test = pickle.load(f)
#test_np = test.toarray()
print('Data loaded.')

print('user_num:',train.shape[0],'item_num:',train.shape[1],'lambda_1:',lambda_1,'lambda_2:',lambda_2,'temp:',temp,'q:',svd_q)

epoch_user = min(train.shape[0], 30000)

adj = scipy_sparse_mat_to_torch_sparse_tensor(train).coalesce().cuda(torch.device(device))

print('Performing SVD...')
svd_u,s,svd_v = torch.svd_lowrank(adj,q=svd_q)
u_mul_s = svd_u @ torch.diag(s)
v_mul_s = svd_v @ torch.diag(s)
del adj
del s
print('SVD done.')

rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)
adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')

test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

loss_list = []
loss_r_list = []
loss_s_list = []
recall_20_x = []
recall_20_y = []
ndcg_20_y = []
recall_40_y = []
ndcg_40_y = []

model = LightGCL(adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T, train_csr, adj_norm, l, temp, lambda_1, dropout, batch_user, device)
#model.load_state_dict(torch.load('saved_model.pt'))
model.cuda(torch.device(device))
optimizer = torch.optim.Adam(model.parameters(),weight_decay=lambda_2,lr=lr)
#optimizer.load_state_dict(torch.load('saved_optim.pt'))

def learning_rate_decay(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']*0.98
        if lr > 0.0005:
            param_group['lr'] = lr
    return lr

current_lr = lr

for epoch in range(epoch_no):
    if (epoch+1)%50 == 0:
        torch.save(model.state_dict(),'saved_model_epoch_'+str(epoch)+'.pt')
        torch.save(optimizer.state_dict(),'saved_optim_epoch_'+str(epoch)+'.pt')

    current_lr = learning_rate_decay(optimizer)
        
    e_users = np.random.permutation(adj_norm.shape[0])[:epoch_user]
    batch_no = int(np.ceil(epoch_user/batch_user))

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    for batch in tqdm(range(batch_no)):
        start = batch*batch_user
        end = min((batch+1)*batch_user,epoch_user)
        batch_users = e_users[start:end]

        # sample pos and neg
        pos = []
        neg = []
        iids = set()
        for i in range(len(batch_users)):
            u = batch_users[i]
            u_interact = train_csr[u].toarray()[0]
            positive_items = np.random.permutation(np.where(u_interact==1)[0])
            negative_items = np.random.permutation(np.where(u_interact==0)[0])
            item_num = min(max_samp,len(positive_items))
            positive_items = positive_items[:item_num]
            negative_items = negative_items[:item_num]
            pos.append(torch.LongTensor(positive_items).cuda(torch.device(device)))
            neg.append(torch.LongTensor(negative_items).cuda(torch.device(device)))
            iids = iids.union(set(positive_items))
            iids = iids.union(set(negative_items))
        iids = torch.LongTensor(list(iids)).cuda(torch.device(device))
        uids = torch.LongTensor(batch_users).cuda(torch.device(device))

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(uids, iids, pos, neg)
        loss.backward()
        optimizer.step()
        #print('batch',batch)

        torch.cuda.empty_cache()

        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

    epoch_loss = epoch_loss/batch_no
    epoch_loss_r = epoch_loss_r/batch_no
    epoch_loss_s = epoch_loss_s/batch_no
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

    if epoch % 3 == 0:  # test every 10 epochs
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_no = int(np.ceil(len(test_uids)/batch_user))

        all_recall_20 = 0
        all_ndcg_20 = 0
        all_recall_40 = 0
        all_ndcg_40 = 0
        for batch in tqdm(range(batch_no)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
            predictions = model(test_uids_input,None,None,None,test=True)
            predictions = np.array(predictions.cpu())

            #top@20
            recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
            #top@40
            recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

            all_recall_20+=recall_20
            all_ndcg_20+=ndcg_20
            all_recall_40+=recall_40
            all_ndcg_40+=ndcg_40
            #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
        print('-------------------------------------------')
        print('Test of epoch',epoch,':','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)
        recall_20_x.append(epoch)
        recall_20_y.append(all_recall_20/batch_no)
        ndcg_20_y.append(all_ndcg_20/batch_no)
        recall_40_y.append(all_recall_40/batch_no)
        ndcg_40_y.append(all_ndcg_40/batch_no)

# final test
test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_no = int(np.ceil(len(test_uids)/batch_user))

all_recall_20 = 0
all_ndcg_20 = 0
all_recall_40 = 0
all_ndcg_40 = 0
for batch in range(batch_no):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).cuda(torch.device(device))
    predictions = model(test_uids_input,None,None,None,test=True)
    predictions = np.array(predictions.cpu())

    #top@20
    recall_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    #top@40
    recall_40, ndcg_40 = metrics(test_uids[start:end],predictions,40,test_labels)

    all_recall_20+=recall_20
    all_ndcg_20+=ndcg_20
    all_recall_40+=recall_40
    all_ndcg_40+=ndcg_40
    #print('batch',batch,'recall@20',recall_20,'ndcg@20',ndcg_20,'recall@40',recall_40,'ndcg@40',ndcg_40)
print('-------------------------------------------')
print('Final test:','Recall@20:',all_recall_20/batch_no,'Ndcg@20:',all_ndcg_20/batch_no,'Recall@40:',all_recall_40/batch_no,'Ndcg@40:',all_ndcg_40/batch_no)

recall_20_x.append('Final')
recall_20_y.append(all_recall_20/batch_no)
ndcg_20_y.append(all_ndcg_20/batch_no)
recall_40_y.append(all_recall_40/batch_no)
ndcg_40_y.append(all_ndcg_40/batch_no)

metric = pd.DataFrame({
    'epoch':recall_20_x,
    'recall@20':recall_20_y,
    'ndcg@20':ndcg_20_y,
    'recall@40':recall_40_y,
    'ndcg@40':ndcg_40_y
})
current_t = time.gmtime()
metric.to_csv('log/result_'+args.data+'_'+time.strftime('%Y-%m-%d_%H_%M_%S',current_t)+'.csv')

torch.save(model.state_dict(),'saved_model/saved_model_'+args.data+'_'+time.strftime('%Y-%m-%d_%H_%M_%S',current_t)+'.pt')
torch.save(optimizer.state_dict(),'saved_model/saved_optim_'+args.data+'_'+time.strftime('%Y-%m-%d_%H_%M_%S',current_t)+'.pt')