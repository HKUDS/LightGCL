import torch
import torch.nn as nn
from utils import sparse_dropout, spmm

class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, dropout, batch_user, device):
        super(LightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.temp = temp
        self.lambda_1 = lambda_1
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user
        self.Ws = nn.ModuleList([W_contrastive(d) for i in range(l)])

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask)
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                self.Z_u_list[layer] = self.act(spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1], self.device))
                self.Z_i_list[layer] = self.act(spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1], self.device))
                
                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = self.act(self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = self.act(self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer] + self.E_u_list[layer-1]
                self.E_i_list[layer] = self.Z_i_list[layer] + self.E_i_list[layer-1]

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            loss_s = 0
            for l in range(1,self.l+1):
                u_mask = (torch.rand(len(uids))>0.5).float().cuda(self.device)

                gnn_u = nn.functional.normalize(self.Z_u_list[l][uids],p=2,dim=1)
                hyper_u = nn.functional.normalize(self.G_u_list[l][uids],p=2,dim=1)
                hyper_u = self.Ws[l-1](hyper_u)
                pos_score = torch.exp((gnn_u*hyper_u).sum(1)/self.temp)
                neg_score = torch.exp(gnn_u @ hyper_u.T/self.temp).sum(1)
                loss_s_u = ((-1 * torch.log(pos_score/(neg_score+1e-8) + 1e-8))*u_mask).sum()
                loss_s = loss_s + loss_s_u

                i_mask = (torch.rand(len(iids))>0.5).float().cuda(self.device)

                gnn_i = nn.functional.normalize(self.Z_i_list[l][iids],p=2,dim=1)
                hyper_i = nn.functional.normalize(self.G_i_list[l][iids],p=2,dim=1)
                hyper_i = self.Ws[l-1](hyper_i)
                pos_score = torch.exp((gnn_i*hyper_i).sum(1)/self.temp)
                neg_score = torch.exp(gnn_i @ hyper_i.T/self.temp).sum(1)
                loss_s_i = ((-1 * torch.log(pos_score/(neg_score+1e-8) + 1e-8))*i_mask).sum()
                loss_s = loss_s + loss_s_i
            
            # bpr loss
            loss_r = 0
            for i in range(len(uids)):
                u = uids[i]
                u_emb = self.E_u[u]
                u_pos = pos[i]
                u_neg = neg[i]
                pos_emb = self.E_i[u_pos]
                neg_emb = self.E_i[u_neg]
                pos_scores = u_emb @ pos_emb.T
                neg_scores = u_emb @ neg_emb.T
                bpr = nn.functional.relu(1-pos_scores+neg_scores)
                loss_r = loss_r + bpr.sum()
            loss_r = loss_r/self.batch_user

            # total loss
            loss = loss_r + self.lambda_1 * loss_s
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, loss_s

class W_contrastive(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(d,d)))

    def forward(self,x):
        return x @ self.W