# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


############################################################################################################################################
############################################################################################################################################

class LScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1,groups=1):

        super(LScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model//groups, h * d_k//groups)    ##(512//2,8*192//2)
        self.fc_k = nn.Linear(d_model//groups, h * d_k//groups)    ##(512//2,8*192//2)
        self.fc_v = nn.Linear(d_model//groups, h * d_v//groups)    ##(512//2,8*64//2)
        self.fc_o = nn.Linear(h * d_v//groups,  d_model//groups)       ##(8*64//2,512//2)
        self.dropout = nn.Dropout(dropout)
        self.groups = groups

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries.view(b_s,nq,self.groups,-1)).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys.view(b_s,nk,self.groups,-1)).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values.view(b_s,nk,self.groups,-1)).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            att = att.masked_fill(attention_mask == 0, float('-inf'))
        att = torch.softmax(att, -1)
        p_attn = att
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out.view(b_s,nq,self.groups,-1)).view(b_s,nq,-1)  # (b_s, nq, d_model)
        return out, p_attn


class LMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, dropout=.1, groups=1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(LMultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = LScaledDotProductAttention(d_model=d_model, groups=groups, d_k=d_k, d_v=d_v, h=h)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.can_be_stateful = can_be_stateful

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        if attention_mask is not None:
            # Same mask applied to all h heads.
            attention_mask = attention_mask.unsqueeze(1)

        out, attn = self.attention(queries, keys, values, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out


############################################################################################################################################
############################################################################################################################################

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LPositionWiseFeedForward(nn.Module):

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, groups=2, identity_map_reordering=False):
        super(LPositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.hiddens=d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff//groups, d_model//groups)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.groups=groups

    def forward(self, input):
        b,l,d = input.size()
        out = self.dropout_2(F.relu(self.fc1(input)))
        out = out.view(b, l, self.groups, self.hiddens // self.groups)
        out=self.fc2(out).view(b,l,d)
        out = self.dropout(out)
        out = self.layer_norm(input + out)
        return out

############################################################################################################################################


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, fc_feats, src, word_feats, attr_feats, seg_feats, boxes_feats, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        x, fusion_v = self.encode(fc_feats, src, word_feats, attr_feats, seg_feats, boxes_feats, src_mask)
        x, attn_e, attn_c = self.decode(x, src_mask, tgt, tgt_mask)
        return x, fusion_v, attn_e, attn_c
    
    def encode(self, fc_feats, src, word_feats, attr_feats, seg_feats, boxes_feats, src_mask):
        #print(self.encoder(self.src_embed(src), src_mask).shape)
        return self.encoder(fc_feats, self.src_embed(src), word_feats, attr_feats, seg_feats, boxes_feats, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        #print(self.tgt_embed(tgt).shape)
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.size = 512
        self.dropout = 0.1
        self.norm = LayerNorm(layer.size)

    def forward(self, fc_feats, att, word_feats, attr_feats, seg_feats, boxes_feats, mask):     ## fc(5,512)
        
        x = att
        fusion_v = x

        for layer in self.layers:
            x = layer(fc_feats, x, word_feats, attr_feats, boxes_feats, mask)  ### x -> att

        return self.norm(x), fusion_v

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, fc_feats, att, word_feats, attr_feats, boxes_feats, mask):

        x = att

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        attn_e = x
        attn_c = x
        return self.norm(x), attn_e, attn_c

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        #print(x.shape)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))   ### masked MSA
        #print(x.shape)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))    ### cross MSA
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

#####################################################################################################################################
#####################################################################################################################################

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LWTransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = LMultiHeadAttention(d_model=512, d_k=64, d_v=64, h=8, dropout=0.1, groups=2)
        ff = LPositionWiseFeedForward(d_model, d_ff, dropout)    ###PositionwiseFeedForward
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), dropout), N_dec),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(LWTransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)  ### >6,m2transformer needs 3 at least
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)   ## 512
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1
        #self.tgt_vocab = tgt_vocab

        self.word_embedding = Embeddings(self.d_model, tgt_vocab)

        self.linear_fc = nn.Linear(2048, self.d_model)  ##2048
        self.linear_fc_1 = nn.Linear(768, self.d_model)  ##2048


        self.model = self.make_model(0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout)

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, att_masks):
        
        #print('(((((((((((((((((((((((((((')
        #print(att_feats.shape)
        fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, seq, att_masks, seq_mask = \
            self._prepare_feature_forward(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, att_masks)
        #print(att_feats.shape)
        memory, attn = self.model.encode(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, att_masks)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks, attn

    def _prepare_feature_forward(self, fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, att_masks=None, seq=None):
        #fc_feats = self.linear_fc(fc_feats)
        #cls_token_feats = self.linear_fc_1(cls_token_feats)

        word_feats = self.word_embedding(word_feats.long())
        attr_feats = self.word_embedding(attr_feats.long())
        seg_feats = self.word_embedding(seg_feats.long())

        att_masks_ = att_masks
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        boxes_feats = self.clip_att(boxes_feats, att_masks_)[0]

        word_feats = self.clip_att(word_feats, att_masks_)[0]
        attr_feats = self.clip_att(attr_feats, att_masks_)[0]
        seg_feats = self.clip_att(seg_feats, att_masks_)[0]
        

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != 0) & (seq.data != 0)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
                #print(att_feats.shape)
                #print('(((((((((((((((((((')
        else:
            seq_mask = None

        return fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, seq, att_masks=None):   ## word_feats
        #print(fc_feats.shape)         ###(5,2048)

        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])

        fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, seq, att_masks, seq_mask = \
            self._prepare_feature_forward(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, att_masks, seq)

        out = self.model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, seq, att_masks, seq_mask)
              
        outputs = self.model.generator(out[0])

        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, mask, state):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out, attn_e, attn_c = self.model.decode(memory, mask,
                               ys, 
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)], attn_e, attn_c
