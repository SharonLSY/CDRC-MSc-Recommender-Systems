import numpy as np
from copy import deepcopy
import random
import pandas as pd
from scipy import sparse
import sys
import matplotlib.pyplot as plt
%matplotlib inline

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from algorithms.recvae.utils import *


def swish_(x):
    return x.mul_(torch.sigmoid(x))

def swish(x):
    return x.mul(torch.sigmoid(x))

def kl(q_distr, p_distr, weights, eps=1e-7):
    mu_q, logvar_q = q_distr
    mu_p, logvar_p = p_distr
    return 0.5 * (((logvar_q.exp() + (mu_q - mu_p).pow(2)) / (logvar_p.exp() + eps) \
                    + logvar_p - logvar_q - 1
                   ).sum(dim=-1) * weights).mean()

def simple_kl(mu_q, logvar_q, logvar_p_scale, norm):
    return (-0.5 * ( (1 + logvar_q #- torch.log(torch.ones(1)*logvar_p_scale) \
                      - mu_q.pow(2)/logvar_p_scale - logvar_q.exp()/logvar_p_scale
                     )
                   ).sum(dim=-1) * norm
           ).mean()

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())

def log_norm_std_pdf(x):
    return -0.5*(np.log(2 * np.pi) + x.pow(2))

class DeterministicDecoder(nn.Linear):
    def __init__(self, *args):
        super(DeterministicDecoder, self).__init__(*args)

    def forward(self, *args):
        output = super(DeterministicDecoder, self).forward(*args)
        return output, 0


class StochasticDecoder(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(StochasticDecoder, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logvar.data.fill_(-2)

    def forward(self, input):
        
        if self.training:
            std = torch.exp(self.logvar)
            a = F.linear(input, self.weight, self.bias)
            eps = torch.randn_like(a)
            b = eps.mul_(torch.sqrt_(F.linear(input * input, std)))
            output = a + b
            
            kl = (-0.5 * (1 + self.logvar - self.weight.pow(2) - self.logvar.exp())).sum(dim=-1).mean() #/ (10)
            return output, kl
        else:
            output = F.linear(input, self.weight, self.bias)
            return output, 0
        
class GaussianMixturePrior(nn.Module):
    def __init__(self, latent_dim, gaussians_number):
        super(GaussianMixturePrior, self).__init__()
        
        self.gaussians_number = gaussians_number
        
        self.mu_prior = nn.Parameter(torch.Tensor(latent_dim, gaussians_number))
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(latent_dim, gaussians_number))
        self.logvar_prior.data.fill_(0)
        
    def forward(self, z):
        density_per_gaussian = log_norm_pdf(x=z[:, :, None],
                                            mu=self.mu_prior[None, ...].detach(),
                                            logvar=self.logvar_prior[None, ...].detach()
                                           ).add(-np.log(self.gaussians_number))
        
      
        return torch.logsumexp(density_per_gaussian, dim=-1)
    

class GaussianMixturePriorWithAprPost(nn.Module):
    def __init__(self, latent_dim, input_count):
        super(GaussianMixturePriorWithAprPost, self).__init__()
        
        self.gaussians_number = 1
        
        self.mu_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(latent_dim, self.gaussians_number))
        self.logvar_uniform_prior.data.fill_(10)
        
        self.user_mu = nn.Embedding(input_count, latent_dim)
        self.user_logvar = nn.Embedding(input_count, latent_dim)
        
    def forward(self, z, idx):
        density_per_gaussian1 = log_norm_pdf(x=z[:, :, None],
                                            mu=self.mu_prior[None, :, :].detach(),
                                            logvar=self.logvar_prior[None, :, :].detach()
                                           ).add(np.log(1/5 - 1/20))
        
        
        density_per_gaussian2 = log_norm_pdf(x=z[:, :, None],
                                            mu=self.user_mu(idx)[:, :, None].detach(),
                                            logvar=self.user_logvar(idx)[:, :, None].detach()
                                           ).add(np.log(4/5 - 1/20))
        
        density_per_gaussian3 = log_norm_pdf(x=z[:, :, None],
                                            mu=self.mu_prior[None, :, :].detach(),
                                            logvar=self.logvar_uniform_prior[None, :, :].detach()
                                           ).add(np.log(1/10))
        
        density_per_gaussian = torch.cat([density_per_gaussian1,
                                          density_per_gaussian2,
                                          density_per_gaussian3], dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)
    
class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, matrix_dim, axis='users'):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(matrix_dim[1], hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        
        self.prior = GaussianMixturePriorWithAprPost(latent_dim, matrix_dim[0])
        self.decoder = DeterministicDecoder(latent_dim, matrix_dim[1])
        
        self.axis = axis
        self.device = torch.device("cuda")


    def encode(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc21(h5), self.fc22(h5)
    
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, user_ratings, user_idx, beta=1, dropout_rate=0.5, calculate_loss=True, mode=None):
        
        if mode == 'pr':
            mu, logvar = self.encode(user_ratings, dropout_rate=dropout_rate)
        elif mode == 'mf':
            mu, logvar = self.encode(user_ratings, dropout_rate=0)
            
        z = self.reparameterize(mu, logvar)
        x_pred, decoder_loss = self.decode(z)
        
        NLL = -(F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
        
        if calculate_loss:
            if mode == 'pr':
                norm = user_ratings.sum(dim=-1)
                KLD = -(self.prior(z, user_idx) - log_norm_pdf(z, mu, logvar)).sum(dim=-1).mul(norm).mean()
                loss = NLL + beta * KLD + decoder_loss
            
            elif mode == 'mf':
                KLD = NLL * 0
                loss = NLL + decoder_loss
            
            return (NLL, KLD), loss
            
        else:
            return x_pred

    def set_embeddings(self, train_data, momentum=0, weight=None):
        istraining = self.training
        self.eval()

        for batch in generate(batch_size=500, device=self.device, data_in=train_data, axis=self.axis):

            user_ratings = batch.get_ratings_to_dev()
            users_idx = batch.get_idx()

            new_user_mu, new_user_logvar = self.encode(user_ratings, 0)

            old_user_mu = self.prior.user_mu.weight.data[users_idx,:].detach()
            old_user_logvar = self.prior.user_logvar.weight.data[users_idx,:].detach()

            if weight:
                old_user_var = torch.exp(old_user_logvar)
                new_user_var = torch.exp(new_user_logvar)

                post_user_var = 1 / (1 / old_user_var + weight / new_user_var)
                post_user_mu = (old_user_mu / old_user_var + weight * new_user_mu / new_user_var) * post_user_var

                self.prior.user_mu.weight.data[users_idx,:] = post_user_mu
                self.prior.user_logvar.weight.data[users_idx,:] = torch.log(post_user_var + new_user_var)
            else:
                self.prior.user_mu.weight.data[users_idx,:] = momentum * old_user_mu + (1-momentum) * new_user_mu
                self.prior.user_logvar.weight.data[users_idx,:] = momentum * old_user_logvar + (1-momentum) * new_user_logvar

        if istraining:
            self.train()
        else:
            self.eval()

def generate(batch_size, device, data_in, data_out=None, axis='users', shuffle=False, samples_perc_per_epoch=1):
    # assert axis in ['users', 'items']
    assert 0 < samples_perc_per_epoch <= 1
    
    # if axis == 'items':
    #     data_in = data_in.T
    #     if data_out is not None:
    #         data_out = data_out.T
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)

class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


class RecVAE(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=50,
                 n_epochs=50, n_enc_epochs=3, n_dec_epochs=1,
                 batch_size=500, lr=5e-4,
                 not_alternating=False,
                 device=torch.device("cuda"),
                 session_key = 'SessionId', item_key = 'ItemId',
                 model=VAE
                 ):
        
        super(RecVAE, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        
        self.n_epochs = n_epochs
        self.n_enc_epochs = n_enc_epochs
        self.n_dec_epochs = n_dec_epochs
        
        self.batch_size = batch_size
        self.lr = lr
        
        self.device = device
        self.not_alternating = not_alternating
        
        self.session_key = session_key
        self.item_key = item_key
        
        self.session_items = []
        self.session = -1
        
        
    def fit( self, data, test=None ):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        data.drop_duplicates(subset=[self.session_key, self.item_key], keep='first', inplace=True, ignore_index=True)
        test.drop_duplicates(subset=[self.session_key, self.item_key], keep='first', inplace=True, ignore_index=True)
        
        full_data = pd.concat((data, test))
        
        full_data.drop_duplicates(subset=[self.session_key, self.item_key], keep='first', inplace=True, ignore_index=True)
        
        #full_data = self.filter_data(full_data,min_uc=5,min_sc=0) 
        
        itemids = full_data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        self.itemidmap2 = pd.Series(index=np.arange(self.n_items), data=itemids)
        self.predvec = np.zeros( (1, self.n_items) )
        self.predvec = torch.from_numpy(self.predvec).to(self.device)
        
        sessionids = full_data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.useridmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        
        # item2id = dict((sid, i) for (i, sid) in enumerate(itemids))
        # profile2id = dict((pid, i) for (i, pid) in enumerate(sessionids))
        
        data_val_tr, data_val_te = self.split_train_test_proportion( test )
        
        def numerize(tp):
            uid = list(map(lambda x: self.useridmap[x], tp[self.session_key]))
            sid = list(map(lambda x: self.itemidmap[x], tp[self.item_key]))
            return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['SessionId', 'ItemId'])
        
        data = numerize(data)
        data_val_tr = numerize(data_val_tr)
        data_val_te = numerize(data_val_te)
        
        # global indexing
        start_idx = 0
        end_idx = len(sessionids) - 1
        
        rows_tr, cols_tr = data_val_tr['SessionId'] - start_idx, data_val_tr['ItemId']
        rows_te, cols_te = data_val_te['SessionId'] - start_idx, data_val_te['ItemId']
        
        mat_val_tr = sparse.csr_matrix((np.ones_like(rows_tr),(rows_tr, cols_tr)), 
                                    dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        mat_val_te = sparse.csr_matrix((np.ones_like(rows_te),(rows_te, cols_te)), 
                                    dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        
        # train
        ones = np.ones( len(data) )
        col_ind = self.itemidmap[ data[self.item_key].values ]
        row_ind = self.useridmap[ data[self.session_key].values ] 
        mat = sparse.csr_matrix((ones, (row_ind, col_ind)), shape=(self.n_sessions, self.n_items))
        
        ###############
        
        self.input_dim = mat.shape
        
        self.vae_model = VAE(self.hidden_dim, self.latent_dim, self.input_dim).to(self.device)
        self.vae_model_best = VAE(self.hidden_dim, self.latent_dim, self.input_dim).to(self.device)
        
        self.decoder_params = set(self.vae_model.decoder.parameters())
        self.embedding_params = set(self.vae_model.prior.user_mu.parameters()) | set(self.vae_model.prior.user_logvar.parameters())
        self.encoder_params = set(self.vae_model.parameters()) - self.decoder_params - self.embedding_params
        
        self.optimizer_encoder = optim.Adam(self.encoder_params, lr=self.lr)
        self.optimizer_decoder = optim.Adam(self.decoder_params, lr=self.lr)
        self.optimizer_embedding = optim.Adam(self.embedding_params, lr=self.lr)
        
        self.train_data = mat
        
        metrics = [{'metric': ndcg, 'k': 100}]
        
        best_ndcg = -np.inf
        ndcgs_tr_pr, ndcgs_tr_mf, ndcgs_va_pr, ndcgs_va_mf = [], [], [], []
        var_param_distance = []
        
        for epoch in range(50):

            self.run(opts=[self.optimizer_encoder], n_epochs=self.n_enc_epochs, mode='pr', beta=0.005)
            self.vae_model.set_embeddings(self.train_data)
            self.run(opts=[self.optimizer_decoder], n_epochs=self.n_dec_epochs, mode='mf', beta=None)
            
            ndcg_ = self.validate(self.train_data, self.train_data, mode='mf', samples_perc_per_epoch=0.01)
            ndcgs_tr_mf.append(ndcg_)
            ndcg_ = self.validate(self.train_data, self.train_data, mode='pr', samples_perc_per_epoch=0.01)
            ndcgs_tr_pr.append(ndcg_)
            ndcg_ = self.validate(mat_val_tr, mat_val_te, mode='pr', samples_perc_per_epoch=1)
            ndcgs_va_pr.append(ndcg_)
            
            # clear_output(True)
            
            i_min = np.array(ndcgs_va_pr).argsort()[-len(ndcgs_va_pr)//2:].min()
        
            print('ndcg', ndcgs_va_pr[-1], ': : :', best_ndcg)
            fig, ax1 = plt.subplots()
            fig.set_size_inches(15,5)
        
            ax1.plot(range(i_min, len(ndcgs_va_pr)), ndcgs_va_pr[i_min:], '+-', label='pr valid')
            ax1.legend(loc='lower right')
            ax1.grid(True)
        
            ax2 = ax1.twinx()
            ax2.plot(range(i_min, len(ndcgs_va_pr)), ndcgs_tr_pr[i_min:], '+:', label='pr train')
            ax2.plot(range(i_min, len(ndcgs_va_pr)), ndcgs_tr_mf[i_min:], 'x:', label='mf train')
            ax2.legend(loc='lower left')
        
            fig.tight_layout()
            plt.ylabel("Validation NDCG@100")
            plt.xlabel("Epochs")
            plt.show()
        
            if ndcg_ > best_ndcg:
                best_ndcg = ndcg_
                torch.save(model.state_dict(), ser_model_fn)
                self.vae_model_best.load_state_dict(deepcopy(self.vae_model.state_dict()))
                
            if ndcg_ < best_ndcg / 2 and epoch > 10:
                break
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        name : int or string
            The session IDs of the event.
        tracks : int list
            The item ID of the event. Must be in the set of item IDs of the training set.
            
        Returns
        --------
        res : pandas.DataFrame
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        
        if session_id != self.session:
            self.session_items = []
            self.session = session_id
            self.predvec = torch.zeros_like(self.predvec)
            #self.predvec[0].fill(0)
        
        if type == 'view':
            self.session_items.append( input_item_id )
            if input_item_id in self.itemidmap:
                self.predvec[0][ self.itemidmap[input_item_id] ] = 1
            
        if skip:
            return
        
        recommendations = self.vae_model(self.predvec.float(), calculate_loss=False, mode='mf').cpu().detach().numpy().flatten()
        
        series = pd.Series( data=recommendations, index=self.itemidmap.index )
        
        return series
        
    def filter_data(self, data, min_uc=5, min_sc=0):
        # Only keep the triplets for items which were clicked on by at least min_sc users. 
        if min_sc > 0:
            itemcount = data[[self.item_key]].groupby(self.item_key).size()
            data = data[data[self.item_key].isin(itemcount.index[itemcount.values >= min_sc])]
        
        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
        if min_uc > 0:
            usercount = data[[self.session_key]].groupby(self.session_key).size()
            data = data[data[self.session_key].isin(usercount.index[usercount.values >= min_uc])]
        
        return data
    
    def split_train_test_proportion(self, data, test_prop=0.2):
        
        data_grouped_by_user = data.groupby( self.session_key )
        tr_list, te_list = list(), list()
    
        np.random.seed(98765)
    
        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)
    
            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
    
                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            else:
                tr_list.append(group)
    
            if i % 1000 == 0:
                print("%d users sampled" % i)
                sys.stdout.flush()
        
        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        
        return data_tr, data_te
    
    def run(self, opts, n_epochs, beta, mode, axis='user'):
        global best_ndcg
        global ndcgs_tr_pr, ndcgs_tr_mf, ndcgs_va_pr, ndcgs_va_mf
        
        for epoch in range(n_epochs):
            self.vae_model.train()
            NLL_loss = 0
            KLD_loss = 0
    
            for batch in generate(batch_size=self.batch_size, device=self.device, axis=axis, data_in=self.train_data, shuffle=True):
                ratings = batch.get_ratings_to_dev()
                idx = batch.get_idx_to_dev()
    
                for optimizer in opts:
                    optimizer.zero_grad()
                    
                (NLL, KLD), loss = self.vae_model(ratings, idx, beta=beta, mode=mode)
                loss.backward()
                
                for optimizer in opts:
                    optimizer.step()
                
                NLL_loss += NLL.item()
                KLD_loss += KLD.item()
                
    
            print('NLL_loss', NLL_loss, 'KLD_loss', KLD_loss)
        
    
    # def evaluate(self, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500):
    #     metrics = deepcopy(metrics)
    #     self.vae_model.eval()
        
    #     for m in metrics:
    #         m['score'] = []
        
    #     for batch in generate(batch_size=self.batch_size,
    #                           device=self.device,
    #                           data_in=data_in,
    #                           data_out=data_out,
    #                           samples_perc_per_epoch=samples_perc_per_epoch
    #                          ):
            
    #         ratings_in = batch.get_ratings_to_dev()
    #         ratings_out = batch.get_ratings(is_out=True)
            
    #         ratings_pred = self.vae_model(ratings_in, calculate_loss=False).cpu().detach().numpy()
            
    #         if not (data_in is data_out):
    #             ratings_pred[batch.get_ratings().nonzero()] = -np.inf
                
    #         for m in metrics:
    #             m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))
    
    #     for m in metrics:
    #         m['score'] = np.concatenate(m['score']).mean()
            
    #     return [x['score'] for x in metrics]
    
    def validate(self, data_1, data_2, mode, axis='users', samples_perc_per_epoch=1):
        self.vae_model.eval()
        ndcg_dist = []
        
        
        for batch in generate(batch_size=self.batch_size,
                              device=self.device,
                              axis=axis,
                              data_in=data_1,
                              data_out=data_2,
                              samples_perc_per_epoch=samples_perc_per_epoch
                             ):
            
            ratings = batch.get_ratings_to_dev()
            idx = batch.get_idx_to_dev()
            ratings_test = batch.get_ratings(is_out=True)
        
            pred_val = self.vae_model(ratings, idx, calculate_loss=False, mode=mode).cpu().detach().numpy()
            
            if not (data_1 is data_2):
                pred_val[batch.get_ratings().nonzero()] = -np.inf
            ndcg_dist.append(ndcg(pred_val, ratings_test))
    
        ndcg_dist = np.concatenate(ndcg_dist)
        if ndcg_dist[~np.isnan(ndcg_dist)] == []:
            return 0
        else:
            return ndcg_dist[~np.isnan(ndcg_dist)].mean()
        
        
        