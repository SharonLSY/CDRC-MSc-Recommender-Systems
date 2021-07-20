import numpy as np
from copy import deepcopy
import random
import pandas as pd
from scipy import sparse
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from algorithms.recvae.utils import *


def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)
    

class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)    
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        
        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)
            
            return (mll, kld), negative_elbo
            
        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
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
                 batch_size=500, beta=None, gamma=0.005, lr=5e-4,
                 not_alternating=False,
                 device=torch.device("cuda:0"),
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
        self.beta = beta
        self.gamma = gamma
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
        
        self.input_dim = mat.shape[1]
        
        self.vae_model = VAE(self.hidden_dim, self.latent_dim, self.input_dim).to(self.device)
        self.vae_model_best = VAE(self.hidden_dim, self.latent_dim, self.input_dim).to(self.device)
        
        self.decoder_params = set(self.vae_model.decoder.parameters())
        self.encoder_params = set(self.vae_model.encoder.parameters())

        self.optimizer_encoder = optim.Adam(self.encoder_params, lr=self.lr)
        self.optimizer_decoder = optim.Adam(self.decoder_params, lr=self.lr)
        
        self.train_data = mat
        
        metrics = [{'metric': ndcg, 'k': 100}]

        best_ndcg = -np.inf
        train_scores, valid_scores = [], []
        
        for epoch in range(self.n_epochs):

            if self.not_alternating:
                self.run(opts=[self.optimizer_encoder, self.optimizer_decoder], n_epochs=1, dropout_rate=0.5)
            else:
                self.run(opts=[self.optimizer_encoder], n_epochs=self.n_enc_epochs, dropout_rate=0.5)
                self.vae_model.update_prior()
                self.run(opts=[self.optimizer_decoder], n_epochs=self.n_dec_epochs, dropout_rate=0)
        
            train_scores.append(
                self.evaluate(self.train_data, self.train_data, metrics, 0.01)[0]
            )
            valid_scores.append(
                self.evaluate(mat_val_tr, mat_val_te, metrics, 1)[0]
            )
            
            if valid_scores[-1] > best_ndcg:
                best_ndcg = valid_scores[-1]
                self.vae_model_best.load_state_dict(deepcopy(self.vae_model.state_dict()))
                
        
            print(f'epoch {epoch} | valid ndcg@100: {valid_scores[-1]:.4f} | ' +
                  f'best valid: {best_ndcg:.4f} | train ndcg@100: {train_scores[-1]:.4f}')
    
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
        
        recommendations = self.vae_model(self.predvec.float(), calculate_loss=False).cpu().detach().numpy().flatten()
        
        series = pd.Series( data=recommendations, index=self.itemidmap.index )
        
        return series

    def clear(self):
        for layer in self.vae_model.children():
           if hasattr(layer, 'reset_parameters'):
               layer.reset_parameters()
        
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
    
    def run(self, opts, n_epochs, dropout_rate):
        self.vae_model.train()
        for epoch in range(n_epochs):
            for batch in generate(batch_size=self.batch_size, device=self.device, data_in=self.train_data, shuffle=True):
                ratings = batch.get_ratings_to_dev()
    
                for optimizer in opts:
                    optimizer.zero_grad()
                    
                _, loss = self.vae_model(ratings, beta=self.beta, gamma=self.gamma, dropout_rate=dropout_rate)
                loss.backward()
                
                for optimizer in opts:
                    optimizer.step()
    
    def evaluate(self, data_in, data_out, metrics, samples_perc_per_epoch=1, batch_size=500):
        metrics = deepcopy(metrics)
        self.vae_model.eval()
        
        for m in metrics:
            m['score'] = []
        
        for batch in generate(batch_size=self.batch_size,
                              device=self.device,
                              data_in=data_in,
                              data_out=data_out,
                              samples_perc_per_epoch=samples_perc_per_epoch
                             ):
            
            ratings_in = batch.get_ratings_to_dev()
            ratings_out = batch.get_ratings(is_out=True)
            
            ratings_pred = self.vae_model(ratings_in, calculate_loss=False).cpu().detach().numpy()
            
            if not (data_in is data_out):
                ratings_pred[batch.get_ratings().nonzero()] = -np.inf
                
            for m in metrics:
                m['score'].append(m['metric'](ratings_pred, ratings_out, k=m['k']))
    
        for m in metrics:
            m['score'] = np.concatenate(m['score']).mean()
            
        return [x['score'] for x in metrics]
        
        
        