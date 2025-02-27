#!/usr/bin/env python
# coding: utf-8

# In[37]:


# import PyTorch package(s)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.utils import clip_grad_norm_

# sklearns + numpy + pandas
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# misc
import random
import numpy as np
import dill as pickle
from   SmilesPE.pretokenizer import atomwise_tokenizer
import math

# import rdkit
from rdkit import Chem
from collections import Counter, OrderedDict
from tqdm.auto import tqdm

# In[20]:


# Example parameters
hidden_size = 512
latent_size = 32
batch_size  = 100
num_epochs  = 25
embedding_dims = 5

trainingFile  = 'BindingDB/preProcessedV1_noHS.csv' # Change this file if you have a new pre-processed file from different notebook
src_col       = 'seq_0'
trg_col       = 'can_smiles'


# In[21]:


def validSMILES(smile):
    m = Chem.MolFromSmiles(smile,sanitize=False)
    if m is None:
       print('invalid SMILES ', smile)
       return False
    else:
      try:
        Chem.SanitizeMol(m)
      except:
        print('invalid chemistry ',smile)
        return False
    return True

def read_data( trainingFile ):
    df = pd.read_csv( trainingFile )
    print ("Input file has %d pairs" % df.shape[0] )
    df.drop_duplicates([ trg_col, src_col ], inplace=True)
    print ("number of de-duped training pairs before sanitization ", df.shape[0])
    validMask = df[trg_col].apply( lambda smi : validSMILES(smi))
    df = df[ validMask ]
    return df[[src_col,trg_col]]


# In[22]:


def smileTokenizer( smi ):
    return atomwise_tokenizer(smi)

def proteinTokenizer(protSequence):
    encoding_data = [ aa.upper() for aa in protSequence]
    return encoding_data     


# In[23]:


class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'


class CharVocab:
    @classmethod
    def from_data(cls, data,  *args, **kwargs):
        chars = set()
        for string in data:
            chars.update(string)
        return cls(chars,  *args, **kwargs)

    
    def __init__(self, chars,  ss=SpecialTokens):
        if (ss.bos in chars) or (ss.eos in chars) or \
                (ss.pad in chars) or (ss.unk in chars):
            raise ValueError('SpecialTokens in chars')

        all_syms = sorted(list(chars)) + [ss.bos, ss.eos, ss.pad, ss.unk]

        self.ss = ss
        self.c2i = {c: i  for i, c in enumerate(all_syms)}
        self.i2c = {i: c for i, c in enumerate(all_syms)}

    def __len__(self):
        return len(self.c2i)

    @property
    def bos(self):
        return self.c2i[self.ss.bos]

    @property
    def eos(self):
        return self.c2i[self.ss.eos]

    @property
    def pad(self):
        return self.c2i[self.ss.pad]

    @property
    def unk(self):
        return self.c2i[self.ss.unk]

    def char2id(self, char):
        if char not in self.c2i:
            return self.unk

        return self.c2i[char]

    def id2char(self, id):
        if id not in self.i2c:
            return self.ss.unk

        return self.i2c[id]

    def string2ids(self, string,  add_bos=False, add_eos=False):
        ids = [self.char2id(c)  for c in string]

        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]

        return ids

    def ids2string(self, ids,  rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        if rem_bos and ids[0] == self.bos:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.eos:
            ids = ids[:-1]

        string = ''.join([self.id2char(id) for id in ids ])

        return string 


# In[24]:


class OneHotVocab(CharVocab):
    def __init__(self,  *args, **kwargs):
        super(OneHotVocab, self).__init__( *args, **kwargs)
        self.vectors = torch.eye(len(self.c2i))


# In[25]:


class Config:
    device      = 'cuda:0'
    seed        = 0
    smile_vocab_save    = 'models/CVAE_DrugDesign_smile_Vocab_v1.pt'
    protein_vocab_save  = 'models/CVAE_DrugDesign_protein_Vocab_v1.pt'
    trainTest   = 0.8
    n_batch     = 64
    n_workers   = 0
    n_last      = 1000
    # Decoder - GRU configs 
    d_d_h = 512
    d_dropout = 0
    d_n_layers = 3
    d_z = 128
    # Encoder - GRU configs 
    q_bidir = False
    q_cell = 'gru'
    q_d_h  = 256
    q_dropout = 0.5
    q_n_layers = 1
    # Learning rate
    lr_end = 0.000300000000000000000003
    lr_n_mult = 1
    lr_n_period = 10
    lr_n_restarts = 10
    lr_start = 0.001 # 0.000300000000000000000003
    # Weight params for KL Divergence
    kl_start = 0
    kl_w_end = 0.05
    kl_w_start = 0
    # model parameters
    model_save  = 'models/cvae_drugdesign_model_25EPOCHS_FIXED.pt'
    history_save = 'history/cvae_drugdesign_history_25EPOCHS_FIXED.csv'
    save_frequency = 1
    hidden_size = 512
    latent_size = 32
    batch_size  = 64
    num_epochs  = 25
    embedding_dims = 5
    clip_grad = 50
    # for sampling the model
    n_samples = 100
    max_len   = 350
    n_batch   = 32
    # Generated samples
    gen_save = 'models/gen_drugdesign_samples_25EPOCHS_FIXED.txt'


# In[26]:


# Read the training datasets and build the SMILES vocabs using CharVocab
corpusDF      = read_data( trainingFile )

# Split them into training and validation datasets

train_data = corpusDF[:int(corpusDF.shape[0]* Config.trainTest) ] 
val_data   = corpusDF[int(corpusDF.shape[0]* Config.trainTest)+1: ] 
device     = torch.device(Config.device)


# In[27]:


# Transform / pre-process the training datasets into one-hot encoding and use it as a vocab and save it
allSMILES     = [ smileTokenizer( smile[0] )     for  smile in corpusDF[[trg_col]].drop_duplicates().values.tolist() ]
allPROTEINS   = [ proteinTokenizer( protein[0] ) for  protein in corpusDF[[src_col]].drop_duplicates().values.tolist() ]

smileVocab   = OneHotVocab.from_data( allSMILES  )
proteinVocab = OneHotVocab.from_data( allPROTEINS ) 

torch.save( smileVocab,   Config.smile_vocab_save )
torch.save( proteinVocab, Config.protein_vocab_save )


# In[28]:


def set_torch_seed_to_all_gens(_):
    seed = torch.initial_seed() % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    
def get_collate_device( model, n_workers):
    return 'cpu' if n_workers > 0 else model.device
    
def get_collate_fn( model, n_workers):
    device = get_collate_device(model, n_workers)
    def collate(data):
        data.sort(key=lambda a: len(a[0]), reverse=True)

        smiles_strings = [string[0] for string in data]
        protein_sequences = [string[1] for string in data]

        dataTensors = pad_sequence([ model.string2tensor( smileTokenizer(smiles), isSMILE=True, device=device) for smiles in smiles_strings ], 
                               batch_first=True, 
                               padding_value=model.pad ) 
        labelTensors = pad_sequence([ model.string2tensor( proteinTokenizer(protein), isSMILE=False,device=device) for protein in protein_sequences ], 
                               batch_first=True, 
                               padding_value=model.proteinPad ) 
        
        return {
                'tokenized_input' : dataTensors,
                'label' : labelTensors 
                }
    return collate
    
def get_vocabulary( data):
    return OneHotVocab.from_data(data)
    
def get_dataloader( model, data,  collate_fn=None, shuffle=True, n_workers=1):
    if collate_fn is None:
        collate_fn = get_collate_fn(model, n_workers)
    return DataLoader(data, 
                      batch_size=Config.n_batch,
                      shuffle=shuffle,
                      num_workers=n_workers, 
                      collate_fn=collate_fn,
                      worker_init_fn=set_torch_seed_to_all_gens
                      if n_workers > 0 else None)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _n_epoch():
    return sum(
        Config.lr_n_period * (Config.lr_n_mult ** i)
        for i in range(Config.lr_n_restarts)
    )


# In[29]:


# Define the conditional variational autoencoder model
class CVAE(nn.Module):
    def __init__(self,
                 smileVocab,
                 proteinVocab,
                 config): # input_size,  hidden_size, latent_size, num_classes,  embedding_size):
        super(CVAE, self).__init__()
        
        self.smileVocab   = smileVocab
        self.proteinVocab = proteinVocab
        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(smileVocab, ss))
        self.proteinPad    = proteinVocab.pad  
        
        # Word embeddings layer
        n_smile_vocab, n_protein_vocab = len(smileVocab) , len(proteinVocab)
        print ('n_smile_vocab  n_protein_vocab ', n_smile_vocab, n_protein_vocab )
        
        self.combo_emb   = nn.Embedding(n_smile_vocab + n_protein_vocab, config.embedding_dims, self.pad)
        self.smile_emb   = nn.Embedding(n_smile_vocab ,  config.embedding_dims, self.pad)
        self.protein_emb = nn.Embedding(n_protein_vocab, config.embedding_dims, self.proteinPad, max_norm = True)
        # self.smile_emb.weight.data.copy_(smileVocab.vectors)
        # self.protein_emb.weight.data.copy_(proteinVocab.vectors)
        
        # Using embedding layer to embed class as the condition to VAE architecture
        # self.embed_cond = nn.Embedding(num_embeddings= n_classes,
        #                                embedding_dim = class_embedding_dim,
        #                                max_norm      = True)
        
        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, config.d_z  )
        self.q_logvar = nn.Linear(q_d_last, config.d_z   )
        
        self.encoder_rnn = nn.GRU(
                config.embedding_dims  , #d_smile_emb + d_protein_emb,
                config.q_d_h,
                num_layers=config.q_n_layers,
                batch_first=True,
                dropout=config.q_dropout if config.q_n_layers > 1 else 0,
                bidirectional=config.q_bidir
            )
        self.decoder_rnn = nn.GRU(
            config.embedding_dims + config.d_z,
            config.d_d_h,
            num_layers=config.d_n_layers,
            batch_first=True,
            dropout=config.d_dropout if config.d_n_layers > 1 else 0
        )

        self.decoder_lat = nn.Linear(config.d_z, config.d_d_h)
        self.decoder_fc  = nn.Linear(config.d_d_h, n_smile_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_lat,
            self.decoder_fc
        ])
        
        self.vae = nn.ModuleList([
            self.combo_emb,
            self.encoder,
            self.decoder
        ])


    @property
    def device(self):
        return next(self.parameters()).device

    def string2tensor(self, string, isSMILE, device='model'):
        if isSMILE:
           ids = self.smileVocab.string2ids(string, add_bos=True, add_eos=True)
        else:
           ids = self.proteinVocab.string2ids(string,  add_bos=False, add_eos=False)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device)
        return tensor

    def tensor2string(self, isSMILE, tensor):
        ids = tensor.tolist()
        if isSMILE:
           string = self.smileVocab.ids2string(ids, rem_bos=True, rem_eos=True)
        else:
           string = self.proteinVocab.ids2string(ids, rem_bos=False, rem_eos=False)
        return string
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, y):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """
        # Encoder: x -> z, kl_loss
        # print ('Running encoder')
        # print ("Shape of X is ", x.shape)
        # print ("Shape of Y is ", y.shape)
        z, kl_loss = self.forward_encoder(x,y)
        
        # Decoder: x, z -> recon_loss
        # print ('Running decoder')
        # print ("Shape of Combined Z is ", z.shape)
        recon_loss = self.forward_decoder(x, y, z)
        return kl_loss, recon_loss


    def forward_encoder(self, x,y):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """
        
        x_emb      = self.smile_emb(x)
        y_emb      = self.protein_emb(y)
        xy_emb     = torch.cat( (x_emb,y_emb) , dim=1 )
        # print ('X emb : ', x_emb.shape )
        # print ('Y emb : ', y_emb.shape )
        # print ('XY emb : ', xy_emb.shape )
        xy_pack = nn.utils.rnn.pack_sequence(xy_emb)
 
        _, h = self.encoder_rnn(xy_pack, None)
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        mu, logvar = self.q_mu(h), self.q_logvar(h)
        # re-parameterization
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps
        #print ('Shape of Z is ' , z.shape )
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        return z, kl_loss

    def forward_decoder(self, x, y, z):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """        
        # print ('Dim X before rnn.pad_sequence ', x.shape )
        # print ('Dim Y before rnn.pad_sequence ', y.shape )
        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True,padding_value=self.pad)
        y = nn.utils.rnn.pad_sequence(y, batch_first=True,padding_value=self.pad)
        # print ('Dim X after rnn.pad_sequence ', x.shape )
        # print ('Dim Y after rnn.pad_sequence ', y.shape )
        x_emb = self.smile_emb(x) 
        y_emb = self.protein_emb(y)
        xy_emb = torch.cat( (x_emb,y_emb) , dim=1 )
        # print ('X emb : ', x_emb.shape )
        # print ('Y emb : ', y_emb.shape )
        # print ('X combined : ', xy_emb.shape )
        
        z_0 = z.unsqueeze(1).repeat(1, xy_emb.size(1), 1)
        #print ('z : ', z_0.shape )
        x_input = torch.cat([xy_emb, z_0], dim=-1)
        #print ('x_input : ', x_input.shape )
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
        
        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        
        output, _ = self.decoder_rnn(x_input, h_0)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        #print ('Dim output ', output.shape )
        y_pred = self.decoder_fc(output)
        #print ('Dim of y ', y_pred.shape )
        recon_loss = F.cross_entropy(
            y_pred[:, :-1].contiguous().view(-1, y_pred.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )
        return recon_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """
        return torch.randn(n_batch, 
                           self.q_mu.out_features,
                           device=self.device)

    def sample(self, n_batch, protein_sequence, max_len=100, z=None, temp=1.0):
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            print('Dim of z ', z.shape)
            z_0 = z.unsqueeze(1)
            print('Dim of z_0 ', z_0.shape)
    
            # Convert protein sequence to tensor
            protein_tensor = self.string2tensor(protein_sequence, isSMILE=False)
            protein_tensor = protein_tensor.unsqueeze(0).repeat(n_batch, 1)
    
            # Initial values
            h = self.decoder_lat(z)
            print('Dim of h ', h.shape)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            print('Dim of h unsqueeze ', h.shape)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            print('Dim of w ', w.shape)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch, max_len)
            x[:, 0] = self.bos
            print('Dim of x ', x.shape)
            print(x)
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
            print(end_pads)
            eos_mask = torch.zeros(n_batch, dtype=torch.uint8, device=self.device)
            print(eos_mask)
    
            # Generating cycle
            for i in range(1, max_len):
                print(f'iterating {i}')
                # Concatenate w and protein_tensor along the sequence dimension
                w_with_protein = torch.cat((w.unsqueeze(1), protein_tensor), dim=1)
                print('Dim of w_with_protein ', w_with_protein.shape)
                x_emb = self.combo_emb(w_with_protein)
                print('Dim of x_embed ', x_emb.shape)
                x_input = torch.cat([x_emb, z_0.squeeze(1).unsqueeze(1).repeat(1, x_emb.size(1), 1)], dim=-1)
                print('Dim of x_input ', x_input.shape)
                o, h = self.decoder_rnn(x_input, h)
                print('Dim of o - RNN ', o.shape)
                y = self.decoder_fc(o.squeeze(1))
                print('Dim of y ', y.shape)
    
                # Apply softmax on the last dimension (across classes)
                y = F.softmax(y / temp, dim=-1).view(-1, y.size(-1))  # Reshape to 2D
                w = torch.multinomial(y, 1).view(n_batch, -1)[:, 0]  # Reshape back to the original batch layout after sampling
                print('w :', w)
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = (~eos_mask & (w == self.eos)).bool()
                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask
    
            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])
            print(new_x)
            return [self.tensor2string(True, i_x) for i_x in new_x]


# In[30]:


# Initialize the CVAE model
model  = CVAE(smileVocab, proteinVocab, Config )

model = model.to(device)


# In[31]:


preProcessed_train_data = [ (i[0],i[1]) for i in zip( train_data['can_smiles'].values,train_data['seq_0'].values ) ]
preProcessed_val_data   = [ (i[0],i[1]) for i in zip( val_data['can_smiles'].values,  val_data['seq_0'].values) ]

train_loader = get_dataloader(model, preProcessed_train_data ,  shuffle=True, n_workers=Config.n_workers)
val_loader   = None if val_data is None else get_dataloader(model, preProcessed_val_data, shuffle=False)


# In[32]:


class CircularBuffer:
    def __init__(self, size):
        self.max_size = size
        self.data = np.zeros(self.max_size)
        self.size = 0
        self.pointer = -1

    def add(self, element):
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size
        self.data[self.pointer] = element
        return element

    def last(self):
        assert self.pointer != -1, "Can't get an element from an empty buffer!"
        return self.data[self.pointer]

    def mean(self):
        if self.size > 0:
            return self.data[:self.size].mean()
        return 0.0


# In[33]:


class KLAnnealer:
    def __init__(self, n_epoch, config):
        self.i_start = config.kl_start
        self.w_start = config.kl_w_start
        self.w_max = config.kl_w_end
        self.n_epoch = n_epoch

        self.inc = (self.w_max - self.w_start) / (self.n_epoch - self.i_start)

    def __call__(self, i):
        k = (i - self.i_start) if i >= self.i_start else 0
        print (f"K for KL Annealing is {k}")
        print (f"{self.w_start + k * self.inc}")
        return self.w_start + k * self.inc


# In[34]:


class CosineAnnealingLRWithRestart(_LRScheduler):
    def __init__(self, optimizer, config):
        self.n_period = config.lr_n_period
        self.n_mult = config.lr_n_mult
        self.lr_end = config.lr_end

        self.current_epoch = 0
        self.t_end = self.n_period

        # Also calls first epoch
        super().__init__(optimizer, -1)

    def get_lr(self):
        return [self.lr_end + (base_lr - self.lr_end) *
                (1 + math.cos(math.pi * self.current_epoch / self.t_end)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end


# In[35]:


def _n_epoch():
    return sum(
        Config.lr_n_period * (Config.lr_n_mult ** i)
        for i in range(Config.lr_n_restarts)
    )

def get_optim_params(model):
    return (p for p in model.vae.parameters() if p.requires_grad)
    
def _train_epoch(model, 
                 epoch, 
                 tqdm_data, 
                 kl_weight, 
                 optimizer=None,
                 use_tqdm=False):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    kl_loss_values    = CircularBuffer(Config.n_last)
    recon_loss_values = CircularBuffer(Config.n_last)
    loss_values       = CircularBuffer(Config.n_last)
    unweighted_loss_values       = CircularBuffer(Config.n_last)


    for input_batch in tqdm_data:
        X = input_batch['tokenized_input'].to(device) 
        Y = input_batch['label'].to(device) 
        # Forward
        kl_loss, recon_loss = model( X, Y )
        loss = kl_weight * kl_loss + recon_loss
        unweighted_loss = kl_loss + recon_loss
        # Backward
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(get_optim_params(model),Config.clip_grad)
            optimizer.step()
        # Log
        kl_loss_values.add(kl_loss.item())
        recon_loss_values.add(recon_loss.item())
        loss_values.add(loss.item())
        unweighted_loss_values.add(unweighted_loss.item())
        lr = (optimizer.param_groups[0]['lr']
              if optimizer is not None
              else 0)

        # Update tqdm
        kl_loss_value = kl_loss_values.mean()
        recon_loss_value = recon_loss_values.mean()
        loss_value = loss_values.mean()
        unweighted_loss_value = unweighted_loss_values.mean()
        postfix = [f'loss={loss_value:.5f}',
                   f'(kl={kl_loss_value:.5f}',
                   f'recon={recon_loss_value:.5f})',
                   f'klw={kl_weight:.5f} lr={lr:.5f}',
                   f'unweightd_loss={unweighted_loss_value:.5f}']
        if use_tqdm:
            tqdm_data.set_postfix_str(' '.join(postfix))

    postfix = {
        'epoch': epoch,
        'kl_weight': kl_weight,
        'lr': lr,
        'kl_loss': kl_loss_value,
        'recon_loss': recon_loss_value,
        'loss': loss_value,
        'unweighted_loss': unweighted_loss_value,
        'mode': 'val' if optimizer is None else 'train'}
    return postfix


# model = model.to(device)

# # Define optimizer and loss function
# # criterion   = cvae_loss
# optimizer   = optim.Adam(model.parameters(), lr=Config.lr_start) #get_optim_params(model), lr=Config.lr_start)
# kl_annealer = KLAnnealer(Config.num_epochs, Config)
# lr_annealer = CosineAnnealingLRWithRestart(optimizer,Config)

# model.zero_grad()
# torch.autograd.set_detect_anomaly(True)

# for epoch in range(Config.num_epochs):
#         print (f"<<< ----- Epoch {epoch} ----- >>>")
#         # Epoch start
#         kl_weight = kl_annealer(epoch)
#         tqdm_data = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch))
#         postfix = _train_epoch(model, epoch, tqdm_data, kl_weight, optimizer)

#         if val_loader is not None:
#             tqdm_data = tqdm(val_loader,
#                              desc='Validation (epoch #{})'.format(epoch))
#             postfix = _train_epoch(model, epoch, tqdm_data, kl_weight)

#         if (Config.model_save is not None) and \
#                 (epoch % Config.save_frequency == 0):
#             model = model.to('cpu')
#             torch.save(model.state_dict(),
#                        Config.model_save[:-3] +
#                        '_{0:03d}.pt'.format(epoch))
#             model = model.to(device)

#         # Epoch end
#         lr_annealer.step()



import json
import pandas
import time


# model = model.to(device)

# # Define optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=Config.lr_start)
# kl_annealer = KLAnnealer(Config.num_epochs, Config)
# lr_annealer = CosineAnnealingLRWithRestart(optimizer, Config)

# model.zero_grad()
# torch.autograd.set_detect_anomaly(True)

# # Initialize history log
# history = []

# epoch_start_time = time.time()
# epoch_end_time = 0

# for epoch in range(Config.num_epochs):
#     print(f"<<< ----- Epoch {epoch} ----- >>>")
    
#     kl_weight = kl_annealer(epoch)
#     tqdm_data = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch))
#     train_postfix = _train_epoch(model, epoch, tqdm_data, kl_weight, optimizer, use_tqdm=True)
#     print(f"Train postfix: {train_postfix}")
    
#     # Update history for training
#     history.append(train_postfix)
    
#     if val_loader is not None:
#         tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
#         val_postfix = _train_epoch(model, epoch, tqdm_data, kl_weight, optimizer=None, use_tqdm=True)  # No optimizer for validation
#         print(f"Val postfix: {val_postfix}")
    
#         # Update history for validation
#         history.append(val_postfix)
    
#     if (Config.model_save is not None) and (epoch % Config.save_frequency == 0):
#         model = model.to('cpu')
#         torch.save(model.state_dict(), Config.model_save[:-3] + '_{0:03d}.pt'.format(epoch))
#         model = model.to(device)
    
#     lr_annealer.step()

#     history_df = pd.DataFrame(history)
#     history_df.to_csv(Config.history_save, index=False)

#     epoch_end_time = time.time()

#     print(f"Time elapsed for epoch {epoch}: {round((epoch_end_time - epoch_start_time) / 60, 3) } minutes")
#     epoch_start_time = epoch_end_time

# print("Training completed and history saved.")

# print ("Training Completed")
# model = model.to('cpu')
# torch.save(model.state_dict(), Config.model_save)

# print ("Model saved into ", Config.model_save )


model = CVAE(smileVocab, proteinVocab, Config)
model.load_state_dict(torch.load(f"models/cvae_drugdesign_model_25EPOCHS_FIXED_010.pt"))
model = model.to(device)
# model.eval()  # Set the model to evaluation mode


protein_sequence = 'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF'  # Example protein sequence

n_batch = 50  # Number of SMILES strings to generate


generated_smiles = model.sample(n_batch, protein_sequence)
print('Generated SMiles:')
print(generated_smiles)

# for smile in generated_smiles:
#     isValid = validSMILES(smile)
#     if isValid:
#         print(f"Valid smiles string generated: {smile}")
#     else:
#         print(f"INVALID SMILES: {smile}")

valid_count = 0
invalid_count = 0

with open("./models/gen_drugdesign_samples_25EPOCHS_FIXED_010.txt", 'w') as file:
    file.write(f"Protein Sequence: {protein_sequence}\n\n")
    file.write("Generated SMILES:\n")
    for smile in generated_smiles:
        isValid = validSMILES(smile)
        if isValid:
            valid_count += 1
            print(f"Valid smiles string generated: {smile}\n")
            file.write(f"VALID!!!: {smile}")
        else:
            invalid_count += 1
            print(f"INVALID SMILES: {smile}")
            file.write(f"invalid: {smile}\n")
    
    file.write(f"\nValid SMILES: {valid_count}\n")
    file.write(f"Invalid SMILES: {invalid_count}\n")
    file.write(f"Valid rate: {valid_count/n_batch}\n")
    file.write(f"Total SMILES: {n_batch}\n")

print(f"Generated samples saved to /models/gen_drugdesign_samples_25EPOCHS_FIXED_010.txt")
print(f"Valid SMILES: {valid_count}")
print(f"Invalid SMILES: {invalid_count}")
print(f"Total SMILES: {n_batch}")
        

#Check