"""
MSTT: Multi-Sparse Token Transformer for Tabular Data.

This implementation introduces a multi-module attention architecture with learned feature queries for tabular
classification.
Parts of the implementation were adapted from:

FT-Transformer:
https://github.com/yandex-research/rtdl-revisiting-models

The Annotated Transformer
Alexander Rush (2018)
https://nlp.seas.harvard.edu/2018/04/03/attention.html

The original implementation was modified to incorporate the proposed MSTT architecture.
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch import Tensor

## ---------------------- TABULAR EMBEDDING  ---------------------- 

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model,
                 device,
                 dropout=0.1,
                 divisor=0.01):
        """
        Positional module to introduce order into continuous features.
        """
        super(PositionalEncoding,self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.register_buffer('div_term', torch.exp(torch.arange(0, d_model, 2) * -(math.log(divisor) / d_model)))

    def forward(self,x):
        batch_size, num_features = x.size()
        # Inicializar el tensor de salida
        pos_enc = torch.zeros(batch_size, num_features, self.d_model, device=self.device)
        for i in range(num_features):
            position = x[:, i].unsqueeze(1)  # (batch_size, 1)
            # (batch_size, len(div_term)) -> apply broadcasting
            pos_enc[:, i, 0::2] = torch.sin(position * self.div_term)
            pos_enc[:, i, 1::2] = torch.cos(position * self.div_term)
        if self.dropout:
            pos_enc = self.dropout(pos_enc)
        return pos_enc


class ContinuousEmbedding(nn.Module):
    def __init__(self, continuous_feat:int, emb_dim:int)-> None:
        '''
        Transform continuous features into embedding representation.
        Parameters:
        - continuous_feat (int): The number of continuous features in the dataset.
        - emb_dim (int): The dimension of the embeddings.
        Returns:
        - Tensor: A tensor of shape (batch_size, num_continuous_features, emb_dim) containing the continuous embeddings.
        '''
        super(ContinuousEmbedding, self).__init__()

        self.weights = nn.Parameter(torch.empty(continuous_feat,emb_dim),requires_grad=True)
        self.bias = nn.Parameter(torch.empty(continuous_feat,emb_dim),requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self)-> None:
        # nn.Layer initialization
        d_rsqrt = self.weights.shape[1] ** -0.5
        nn.init.uniform_(self.weights, -d_rsqrt, d_rsqrt)
        nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self,x:Tensor)-> Tensor:
        x = x.unsqueeze(-1)*self.weights
        x = x + self.bias
        return x
        

class CategoricalEmbedding(nn.Module):
    def __init__(self, 
                 cat_cardinalities:list, 
                 emb_dim:int, 
                 bias_st:bool)-> None:
        '''
        Transform categorical features into embedding representation.
        Parameters:
        - cat_cardinalities (list[int]): A list where each element represents the number of unique values in a categorical feature.
        - emb_dim (int): The dimension of the embeddings.
        - bias_st (bool): If True, a trainable vector is added to the embedding of every feature, independent of its value.
        Returns:
        - Tensor: A tensor of shape (batch_size, num_categorical_features, emb_dim) containing the categorical embeddings.
        '''
        super(CategoricalEmbedding, self).__init__()

        self.bias_st = bias_st
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) for num_categories in cat_cardinalities])
        # Bias
        self.bias = nn.Parameter(torch.empty(len(cat_cardinalities),emb_dim),requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self)-> None:
        d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
        for m in self.embeddings:
            nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)
        if self.bias_st is True:
            nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

    def forward(self,x:Tensor)-> Tensor:
        x = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.stack(x, dim=1)
        if self.bias_st is True:
            x = x + self.bias
        return x


class TabularEmbedding(nn.Module):
    def __init__(self, 
                 continuous_feat:int, 
                 emb_dim:int, 
                 bias_st_cat:bool, 
                 device,
                 cat_cardinalities:list=[], 
                 divisor:float=0.01, 
                 positional_enc_st:bool=True)-> None:
        """
        Transforms tabular features into embeddings.
        Parameters:
        - cat_cardinalities (list[int]): A list where each element represents the number of unique values in a categorical fe
        - continuous_feat (int): The number of continuous features in the dataset.
        - emb_dim (int): The dimension of the embeddings.
        - bias_st (bool): If True, a trainable vector is added to the embedding of every feature, independent of its value.
        - divisor (float): Value of divisor in the positional encoding
        - positional_enc_st (bool): When True, the embedding of each continuous feature is combined with a positional encoding vector.
        """
        super(TabularEmbedding, self).__init__()

        self.positional_enc_st = positional_enc_st

        if len(cat_cardinalities) != 0:
            self.categorical_emb = CategoricalEmbedding(cat_cardinalities=cat_cardinalities,
                                                    emb_dim=emb_dim, bias_st=bias_st_cat)
        if continuous_feat != 0:
            self.continuous_feat = ContinuousEmbedding(continuous_feat=continuous_feat, emb_dim=emb_dim)

        if positional_enc_st:
            self.positional_encoding = PositionalEncoding(d_model=emb_dim, divisor=divisor, device=device)

    def forward(self, num_inputs=None, cat_inputs=None)-> Tensor:
        """
        Processes numerical and categorical features.
        Parameters:
        - cat_inputs (Tensor): A tensor containing categorical features with shape (batch_size, num_categorical_features).
        - num_inputs (Tensor): A tensor containing numerical features with shape (batch_size, num_numeric_features).
        Returns:
        - Tensor: A tensor representing the combined embedding with shape (batch_size, total_features, emb_dim).
        """
        if  num_inputs != None:
            num_embedded = self.continuous_feat(num_inputs)

            if self.positional_enc_st:
                num_positional_enc = self.positional_encoding(num_inputs)
                num_embedded = num_embedded + num_positional_enc

            if  cat_inputs!= None:
                cat_embedded = self.categorical_emb(cat_inputs)
                embeddings = torch.cat([num_embedded,cat_embedded], dim=1)
                return embeddings
            else:
                return num_embedded
        else:
            cat_embedded = self.categorical_emb(cat_inputs)
            return cat_embedded

## ---------------------- MULTI-HEAD MODULE ATTENTION  ---------------------- 

class _ReGLU(nn.Module):
    '''
    Gated activation function that combines a linear transformation with a 
    ReLU activation to control how information flows through a neural network.
    '''
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError(
                'For the ReGLU activation, the last input dimension'
                f' must be a multiple of 2, however: {x.shape[-1]=}'
            )
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class MHAttention(nn.Module):
    """
    Vectorized attention where:
      - kv: (B, S_kv, D) shared for all modules
      - q_list: list of N tensors (B, S_qi, D) with variable S_qi
    Each module has independent weights stored as (N, D, D), (N, D) for bias.
    """

    def __init__(self, 
                 num_modules: int, 
                 num_heads: int, 
                 d_model: int, 
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.N = num_modules
        self.H = num_heads
        self.D = d_model
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        # Per-module weights
        self.W_k = nn.Parameter(torch.empty(self.N, d_model, d_model))
        self.b_k = nn.Parameter(torch.zeros(self.N, d_model))

        self.W_v = nn.Parameter(torch.empty(self.N, d_model, d_model))
        self.b_v = nn.Parameter(torch.zeros(self.N, d_model))

        self.W_o = nn.Parameter(torch.empty(self.N, d_model, d_model))
        self.b_o = nn.Parameter(torch.zeros(self.N, d_model))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier for weight tensors (applies per-matrix)
        for W in [self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(W)
        # biases to zero
        for b in [self.b_k, self.b_v, self.b_o]:
            nn.init.zeros_(b)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, not_q_mask: bool, q_mask=None):
        """
        q_list: list length N of tensors (B, F_qi, D)
        kv: (B, F_kv, D)  (shared K,V for all modules) OR (N, B, F_kv, D) if different per module
        Returns:
          outs_list: list length N of tensors (B, F_qi, D) -- outputs per module (unpadded)
        """
        device = kv.device
        B = kv.size(0)
        F_kv = kv.size(1)
        F_q = q.size(2)

        # Projections
        # Create a Big W
        W_big_k = self.W_k.reshape(self.N, self.D, self.D).permute(1, 0, 2).reshape(self.D,self.N*self.D)
        W_big_v = self.W_v.reshape(self.N, self.D, self.D).permute(1, 0, 2).reshape(self.D,self.N*self.D)
        b_big_k = self.b_k.reshape(self.N*self.D)
        b_big_v = self.b_v.reshape(self.N*self.D)
        kv_flat = kv.reshape(B * F_kv, self.D)

        K_proj = kv_flat @ W_big_k + b_big_k
        K_proj = K_proj.reshape(B,F_kv,self.N,self.D).permute(2,0,1,3) # (B,S,N,D) to (N,B,S,D)
        K_proj = torch.einsum('btd,ndo->nbto', kv, self.W_k) +  self.b_k[:, None, None, :]
        V_proj = kv_flat @ W_big_v + b_big_v
        V_proj = V_proj.reshape(B,F_kv,self.N,self.D).permute(2,0,1,3) # (B,S,N,D) to (N,B,S,D)
        V_proj = torch.einsum('btd,ndo->nbto', kv, self.W_v) +  self.b_v[:, None, None, :]

        # Split heads
        Qh = q.view(self.N, B, self.H, -1, self.d_k)
        # For KV (N, B, H, F_kv, d_k)
        Kh = K_proj.contiguous().view(self.N, B, self.H, F_kv, self.d_k)
        Vh = V_proj.contiguous().view(self.N, B, self.H, F_kv, self.d_k)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_k)

        if not_q_mask == False:
            q_mask_exp = q_mask.view(self.N, B, 1, -1, 1).expand(-1, -1, 1, -1, F_kv)  # (N,B,1,S_q_max,S_kv)
            # For invalid Q positions, set scores to -inf so softmax -> 0
            scores = scores.masked_fill(~q_mask_exp, float('-inf'))
            # Detect rows donde todo es -inf
            mask_all_inf = torch.isinf(scores).all(dim=-1, keepdim=True)
            # Reemplazar por ceros (o -1e9, pero NO -inf)
            scores = torch.where(mask_all_inf, torch.zeros_like(scores), scores)

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)  # (N,B,H,S_q_max,S_kv)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # Dot product attention and value
        output = torch.matmul(attn, Vh)

        if not_q_mask == False:
            # Zero-out context rows corresponding to padded Q tokens
            q_mask_out = q_mask.view(self.N, B, 1, -1, 1).expand_as(output)  # (N,B,H,S_q_max,d_k)
            output = output * q_mask_out.to(output.dtype)

        # Combine heads -> (N, B, F_q_max, D)
        output = output.permute(0, 1, 3, 2, 4).contiguous()  # (N,B,F_q_max,H,d_k)
        output = output.view(self.N, B, -1, self.D)

        # Last projection
        out = torch.einsum('nbfd,ndo->nbfo', output, self.W_o) + self.b_o.view(self.N, 1, 1, self.D)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # Calculate mean and variance across the last dimension(s)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta

class PositionalWiseFeedForward(nn.Module):
    def __init__(self,
                 num_modules,
                 model_dim,
                 hidden_dim,
                 dropout):
        super().__init__()
        self.N = num_modules
        self.D = model_dim
        self.H = hidden_dim
        #Linear1 (N, D, H)
        self.W1 = nn.Parameter(torch.empty(num_modules, model_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_modules, hidden_dim))
        #Linear2 (N, H, D)
        self.W2 = nn.Parameter(torch.empty(num_modules, model_dim, model_dim))
        self.b2 = nn.Parameter(torch.zeros(num_modules, model_dim))
        self.dropout = nn.Dropout(dropout)
        self.reglu = _ReGLU()
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier for weight tensors (applies per-matrix)
        for W in [self.W1, self.W2]:
            nn.init.xavier_uniform_(W)
        # biases to zero
        for b in [self.b1, self.b2]:
            nn.init.zeros_(b)

    def forward(self,x):
        x =  torch.einsum("nbfd, ndh -> nbfh", x, self.W1) + self.b1.view(self.N, 1, 1, self.H)
        x = self.reglu(x)
        x = self.dropout(x)
        x = torch.einsum("nbfd,ndd->nbfd", x, self.W2) + self.b2.view(self.N, 1, 1, self.D)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 num_heads:int, 
                 d_model:int, 
                 hidden_size: int, 
                 num_modules: int,
                 pos_norm:bool, 
                 device,
                 num_learned_features:list,
                 attn_dropout:float=0.2, 
                 ffn_dropout:float=0.3,
                 positional_wise:bool=True)-> None:

        super(TransformerEncoder, self).__init__()

        self.positional_wise = positional_wise
        self.num_learned_features = num_learned_features
        self.pos_norm = pos_norm
        self.D = d_model
        self.N = num_modules
        self.device = device
        self.mha = MHAttention(num_modules, num_heads, d_model, dropout = attn_dropout)

        self.norm1_kv = LayerNorm(d_model)
        self.norm1_q = LayerNorm(d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        if positional_wise:
            self.ffn = PositionalWiseFeedForward(num_modules,d_model,hidden_size,ffn_dropout)


    def add_pad_q(self,q_list: List[torch.Tensor], lengths:List[int],pad_value: float = 0.0):
        """
        q_list: list of tensors [ (Batch, num_feat_q, d_model), ... ] Num modules
        returns:
        q_padded: (N, B, F_q_max, D)
        q_mask: (N, B, f_q_max) boolean -> True where token is valid
        """
        B = q_list[0].size(0) # number of batches
        F_q_max = max(lengths)# number of maximum learnable feature in layer
        q_padded = q_list[0].new_zeros((self.N, B, F_q_max, self.D)).fill_(pad_value) # create tensor of zeros
        q_mask = torch.zeros((self.N, B, F_q_max), dtype=torch.bool, device=q_list[0].device) # create tensor of False

        # Mask values if number of learnable features is less than maximum learnable feature in layer
        for i, q in enumerate(q_list):
            L = q.size(1)
            q_padded[i, :, :L, :] = q
            q_mask[i, :, :L] = True
        return q_padded, q_mask


    def _forward_prenorm(self, Q: torch.Tensor, KV: torch.Tensor, not_q_mask: bool,q_mask):
        q_norm = self.norm1_q(Q)
        kv_norm = self.norm1_kv(KV)
        out = self.mha(q_norm,kv_norm,not_q_mask,q_mask)
        #out = out + Q
        if self.positional_wise:
            ffn_out = self.ffn(self.norm2(out))
            out = out + ffn_out
            return out
        return out

    def _forward_posnorm(self, Q: torch.Tensor, KV: torch.Tensor, not_q_mask: bool,q_mask):
        out = self.mha(Q,KV,not_q_mask,q_mask)
        out = self.norm1_q(out)
        if self.positional_wise:
            out = self.norm2(out + self.ffn(out))
            return out
        return out

    def forward(self, q_list: List[torch.Tensor], KV: torch.Tensor):
        not_q_mask = len(set(self.num_learned_features)) == 1
        if not_q_mask:
            q_stack = torch.stack(q_list, dim=0)
            q_mask = None
        else:
            q_stack, q_mask = self.add_pad_q(q_list,self.num_learned_features)
            q_mask = q_mask.to(self.device)
        q_stack = q_stack.to(self.device)
        if self.pos_norm:
            out = self._forward_posnorm(q_stack,KV,not_q_mask,q_mask)
        else:
            out = self._forward_prenorm(q_stack,KV,not_q_mask,q_mask)
        return out


class ParamModule(nn.Module):
    '''
    Generate a trainable vector
    Parameters:
        - num_feat (int): Number of "trainable features"
        - dim (int): Dimension of the vector
    '''
    def __init__(self, num_feat,dim):
        super(ParamModule, self).__init__()
        self.num_feat = num_feat
        self.dim = dim
        self.param = nn.Parameter(torch.empty(1,num_feat, dim),requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self)-> None:
        # nn.Layer initialization
        d_rsqrt = self.param.shape[1] ** -0.5
        nn.init.uniform_(self.param, -d_rsqrt, d_rsqrt)

    def forward(self,x):
        B = x.size(0)
        return self.param.expand(B, -1, -1)


class MHAlayer(nn.Module):
    """
    Implements an MSTT layer consisting of multi-module attention,
    layer normalization, residual connections, and a position-wise
    feed-forward network
    """
    def __init__(self,
                 d_model:int,
                 num_learned_features:list,
                 num_heads:int,
                 hidden_dim_attmlp:int,
                 pos_norm:bool,
                 device):
        super().__init__()
        self.device = device
        self.feature_layer = nn.ModuleList([ParamModule(learnfeat, d_model) for learnfeat in num_learned_features])
        self.num_modules = len(self.feature_layer)
        self.mha = TransformerEncoder(num_heads=num_heads, d_model=d_model, hidden_size=hidden_dim_attmlp,
                                      num_modules=self.num_modules, pos_norm=pos_norm, device=device,
                                      num_learned_features=num_learned_features,attn_dropout=0.2,
                                      ffn_dropout=0.3, positional_wise=True)

    def forward(self,embeddings):
        feature_learnable = [module(embeddings) for module in self.feature_layer] #Crea lista de parametrosentrenables
        out = self.mha(feature_learnable,embeddings)
        outs = []
        lengths = [q.size(1) for q in feature_learnable]
        for i, L in enumerate(lengths):
            outs.append(out[i, :, :L, :].contiguous())
        new_emb = torch.cat(outs,dim=1)
        return new_emb


## ---------------------- MLP FOR CLASSIFICATION ---------------------- 
class MLP(nn.Module):
    '''
    MultiLayer Neural Network for classification
    '''
    def __init__(self,input_dims:int,
                 hidden_dim:int,
                 output_dim:int):
        super(MLP,self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            #nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self,x):
        output = self.mlp(x)
        return output

## ---------------------- MSTT model ---------------------- 

class MSTT(nn.Module):
    """
    Multi-Module Transformer for Tabular Data (MSTT).
    Parameters:
    - d_model (int): Dimensionality of the model's feature representations.
    - cat_cardinalities (list[int]): Number of unique categories for each categorical feature.
    - num_features (int): Number of numerical features.
    - num_learned_features, (list[list[int]]):  Nested list specifying the number 
      of attention modules in each MSTT layer and the number of learnable features 
      assigned to each module.
    - num_heads (int): Number of attention heads in each attention module.
    - hidden_dim_attmlp (int): Hidden dimension of the position-wise feed-forward 
      network within the attention blocks.
    - hidden_dim_supmlp (int): Hidden dimension of the classification MLP.
    - output_dim_supmlp (int): Number of output units in the classification MLP, 
      corresponding to the number of target classes.
    - device (torch.device or str): Device on which the model is executed (CPU or GPU).
    - pos_norm (bool): If True, applies post-layer normalization; otherwise,
      applies pre-layer normalization.
    - positional_enc_status (bool): If True, applies positional encoding to 
      numerical features.
    - divisor (float): Scaling factor used in the positional encoding of numerical
      features when positional_enc_status is True.
    - bias_st_cat (bool) If True, includes a learnable bias in the categorical feature embeddings.
    """
    def __init__(self, 
                 d_model:int,  
                 cat_cardinalities:list, 
                 num_features:int, 
                 num_learned_features:list,
                 num_heads:int,
                 hidden_dim_attmlp:int, 
                 hidden_dim_supmlp:int,
                 output_dim_supmlp:int,
                 device,
                 divisor=None,
                 pos_norm=False,
                 positional_enc_status=False,
                 bias_st_cat=True):
        
        super(MSTT,self).__init__()

        self.d_model = d_model
        self.num_features = num_features
        self.num_learned_features = num_learned_features
        self.num_deep_mha = len(num_learned_features) #number of mhas layers
        self.num_mha_per_layer = [len(i) for i in self.num_learned_features]
        #self.sum_learned_features = [sum(i) for i in self.num_learned_features]
        #input_dims = [[d_model*feat for feat in learnfeat] for learnfeat in num_learned_features]
        #output_dims = [sum(input) for input in input_dims]

        self.embedding_layer = TabularEmbedding(continuous_feat = num_features - len(cat_cardinalities),
                                                cat_cardinalities = cat_cardinalities, 
                                                bias_st_cat = bias_st_cat,
                                                emb_dim = d_model, 
                                                device = device,
                                                divisor = divisor, 
                                                positional_enc_st = positional_enc_status)

        self.blocks = nn.ModuleList([MHAlayer(d_model = d_model, 
                                              num_learned_features = learn_feat,
                                              num_heads = num_heads, 
                                              hidden_dim_attmlp = hidden_dim_attmlp,
                                              pos_norm = pos_norm, 
                                              device = device) for learn_feat in num_learned_features])

        # Aggregator
        self.attention_aggregator = TransformerEncoder(num_heads, 
                                                       d_model, 
                                                       hidden_size = None, 
                                                       num_modules = 1,
                                                       num_learned_features = [1], 
                                                       device = device,
                                                       attn_dropout = 0.2, 
                                                       ffn_dropout = 0.1,
                                                       positional_wise = False , 
                                                       pos_norm = pos_norm)

        self.aggregator_q = ParamModule(1, d_model)

        #MLP classification
        self.mlp_class = MLP(input_dims = d_model,
                             hidden_dim = hidden_dim_supmlp,
                             output_dim = output_dim_supmlp)


    def forward(self, data_num, data_cat):
        embeddings = self.embedding_layer(data_num, data_cat)
        for i, block in enumerate(self.blocks):
            if i == 0:
                input_tensor = embeddings
            else:
                input_tensor = embedd_new
            embedd_new = block(input_tensor)

        # ---- Classification 
        output_agg = self.attention_aggregator([self.aggregator_q(embeddings)], embedd_new).squeeze()
        output = self.mlp_class(output_agg)
        return output


