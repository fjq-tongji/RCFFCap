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

################################################################################
################################################################################
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, kernel):
        modules = [
            nn.Conv1d(in_channels, out_channels, kernel, bias=False)   ## padding=(kernel - 1) // 2
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPP(nn.Module):
    def __init__(self, in_channels=512, atrous_rates=[1, 1, 1]):
        super(ASPP, self).__init__()
        out_channels = 512
        modules = []

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1, 1))
        modules.append(ASPPConv(in_channels, out_channels, rate1, 3))
        #modules.append(ASPPConv(in_channels, out_channels, rate2, 5))
        #modules.append(ASPPConv(in_channels, out_channels, rate3, 7))

        self.convs = nn.ModuleList(modules)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, mask=None):
        x1 = x
        x = x.permute(0, 2, 1).contiguous()  ###(5,B,512) -> (5,512,B)

        res = []
        for conv in self.convs:
            res.append(conv(x))    ###(5,512,B) (5,512,B-2) (5,512,B-4) (5,512,B-6)

        res = torch.cat(res, dim=-1)     ## (5,512,4B-12)
        res = res.permute(0, 2, 1).contiguous()   ## (5,4B-12,512)
        res = res.mean(1).unsqueeze(1)              ## (5,1,512)
        res = F.softmax(res)             ## (5,1,512)
        res = res * x1 + x1              ## (5,B,512)

        return res

################################################################################
################################################################################


##########################################################################

def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value

    #attention weights
    scaled_dot = torch.matmul(w_q,w_k)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

    #w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix
    w_a = scaled_dot
    #w_a = scaled_dot.view(N,N)

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn,w_v)
    return output, w_mn

class BoxMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding = trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        #matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True),8)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = input_query.size(0)

        relative_geometry_embeddings = boxrelationalembedding(input_box, trignometric_embedding= self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1,self.dim_g)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head),1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        if self.legacy_extra_skip:
            x = input_value + x

        return self.linears[-1](x)
#####################################################################################################################################
#####################################################################################################################################


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
        x, attn_1 = self.encode(fc_feats, src, word_feats, attr_feats, seg_feats, boxes_feats, src_mask)
        x, attn = self.decode(x, src_mask, tgt, tgt_mask)

        return x, attn_1, attn
    
    def encode(self, fc_feats, src, word_feats, attr_feats, seg_feats, boxes_feats, src_mask):
        #print(self.encoder(self.src_embed(src), src_mask).shape)
        return self.encoder(fc_feats, self.src_embed(src), word_feats, attr_feats, seg_feats, boxes_feats, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):

        x, attn = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return x, attn

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
        self.norm = LayerNorm(layer.size)

        self.linear = nn.Linear(1024, 512)
        self.linear_1 = nn.Linear(1024, 512)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.cpm = ASPP()

    def forward(self, fc_feats, att, word_feats, attr_feats, seg_feats, boxes_feats, mask):     ## fc(5,512)

        ##############################################################
        ##############################################################
        #attr_word = torch.cat([att,seg_feats], dim=1)  ## (5,B1+B2+B3,512)
        #attr_word = attr_word.mean(1).unsqueeze(1)    ##self.gap(attr_word)      #attr_word.mean(1).unsqueeze(1)     ## (5,1,512)
        #alpha = F.sigmoid(attr_word)
        #x = alpha * att + att  ## (5,B,512)
        #fusion_v = x
        x = att
        ##############################################################

        attention_1 = []
        for layer in self.layers:
            x, attn = layer(fc_feats, x, word_feats, attr_feats, boxes_feats, mask)  ### x -> att
            attention_1.append(attn)

        x = self.cpm(x, mask)

        return self.norm(x), attention_1[-1]

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
        x_yuan = x
        if len(sublayer(self.norm(x_yuan))) > 1:          ### MSA
            x = x + self.dropout(sublayer(self.norm(x))[0])
            attn = sublayer(self.norm(x_yuan))[1]
            return x, attn
        else:
            x = x + self.dropout(sublayer(self.norm(x)))
            return x

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

        x, attn = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)

        return x, attn

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        alpha = []
        for layer in self.layers:
            x, attn = layer(x, memory, src_mask, tgt_mask)
            alpha.append(attn)

        return self.norm(x), alpha[-1]      ## 保存了四个解码器层的交叉注意力层的atten(L,B)


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
        x, attn_1 = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))   ### masked MSA
        x, attn_2 = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))    ### cross MSA
        x = self.sublayer[2](x, self.feed_forward)

        return x, attn_text


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

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
        x, attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), attn


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

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

class TransformerModel_1(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        bbox_attn = BoxMultiHeadedAttention(h, d_model)
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(bbox_attn), c(ff), dropout), N_enc),
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
        super(TransformerModel_1, self).__init__(opt)
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
        fc_feats = self.linear_fc(fc_feats)
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

        out, attn, attn_1 = self.model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, seq, att_masks, seq_mask)

        outputs = self.model.generator(out)

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


        out, attn = self.model.decode(memory, mask,
                               ys, 
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        #print(out.shape)
        #print(attn.shape)
        return out[:, -1], [ys.unsqueeze(0)], attn


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#fc_feats = torch.randn(5, 2048).to(device)
#att_feats = torch.randn(5, 30, 2048).to(device)    ### B=30
#word_feats = torch.randn(5, 30).to(device)
#attr_feats = torch.randn(5, 30).to(device)
#seg_feats = torch.randn(5, 15).to(device)
#boxes_feats = torch.randn(5, 30, 4).to(device)
#seq = torch.randn(5, 18).to(device)                ## L=18
#att_mask = torch.randn(5, 61).to(device)
#model = TransformerModel_1().to(device)

#print(model(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, seq, att_mask).shape)

#macs, params = profile(model, inputs=(fc_feats, att_feats, word_feats, attr_feats, seg_feats, boxes_feats, seq, att_mask,))   ##verbose=False
#print('The number of MACs is %s'%(macs/1e9))   ##### MB
#print('The number of params is %s'%(params/1e6))   ##### MB
