import os,torch,re,math
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from .load_dataset import seqItem2id, vectorize


def split_input(raw_input):
    name = re.split('[\s]', raw_input, 1)[0][1:]
    seqs = re.split('[\\n\\t]', raw_input)[1:]
    seq = ''.join(seqs)
    seq = re.sub('\s', '', seq)
    return name, seq


def norm_seq(seq, max_len=600):
    """ Norm seq length to n*max_len, padding with '*' """
    return seq+'*'*(max_len-(len(seq)%max_len))


def emb_seq(seq, seq_type='protein', max_len=600):
    """ Encode seq,  seq -> tensor [bs * max_len * emb_dim] """
    seq = norm_seq(seq)
    tokenized_seq = [seqItem2id(i, seq_type) for i in seq]
    embedding=nn.Embedding.from_pretrained(torch.tensor(vectorize(emb_type='onehot',seq_type='protein')))
    embed_seq=embedding(torch.LongTensor(tokenized_seq))
    n=int(embed_seq.shape[0]/max_len)
    embed_seq=torch.reshape(embed_seq,(n,max_len,embedding.embedding_dim))
    return embed_seq  # -> [n * max_len * emb_dim]


def load_models(model_path, device):
    """ Load multi saved models """
    models = []
    for roots, dirs, files in os.walk(model_path):
        for file in files:
            if'.pkl'in file:
                # print(file)
                model_pkl = os.path.join(roots, file)
                model = torch.load(model_pkl,map_location=device)
                model.eval()
                models.append(model)
    return models


def predict_essentiality(seq, model_path, device):
    """ Cal essentiality and seq attention for given seq """
    X = emb_seq(seq)  # [n * max_len * emb_dim]
#     print(X.shape)
    models = load_models(model_path, device)
    pred_scores, attn_scores = [], .0
    cnt = 0
    for model in models:
        cnt += 1
        with torch.no_grad():
            X_var = torch.autograd.Variable(X.to(device).float())
        output = model(X_var)
        attn_score = cal_attn(X_var, model)
        pred_scores.append(output[0].detach().numpy())
        attn_scores += attn_score
        
#     print(pred_scores)
    pred_ess = sum(pred_scores)/cnt
    avg_attn = attn_scores/cnt
    return pred_ess, avg_attn


def cal_attn(X, model, num_head=3, kernel_size=3):
    # attention score before softmax
    seq_len = int(torch.sum(torch.sum(torch.sum(X, dim=-1), dim=-1), dim=-1))
    embed_dim = X.shape[-1]
    X = X.view(1, -1, embed_dim)
    X = X[:, :seq_len, :]
    # textCNN
    cnn_in = X.permute(0, 2, 1)
    cnn_w = model.state_dict()['textCNN.weight']
    cnn_b = model.state_dict()['textCNN.bias']
    cnn_out = F.conv1d(cnn_in, cnn_w, cnn_b, padding='same')
    # Res
    res_out = X + cnn_out.permute(0, 2, 1)
    # multiheadAttn
    head_dim = embed_dim // num_head
    w = model.state_dict()['multiAttn.in_proj_weight']
    b = model.state_dict()['multiAttn.in_proj_bias']
    w_q, w_k, w_v = w.chunk(3)
    b_q, b_k, b_v = b.chunk(3)
    q, k, v = F.linear(res_out, w_q, b_q), F.linear(res_out, w_k, b_k), F.linear(res_out, w_v, b_v)
    q = q.contiguous().view(-1, num_head, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, num_head, head_dim).transpose(0, 1)
    q = q / math.sqrt(q.shape[-1])
    attn = torch.bmm(q, k.transpose(-2, -1))
    attn = attn.transpose(0, 1).contiguous().view(num_head, seq_len, seq_len)
    attn = attn.mean(dim=0)   # => [seq_len, seq_len]
    attn = np.array(attn.sum(axis=-1)/seq_len)    
    
    aa_attn = np.zeros_like(attn)
    for i in range(attn.shape[0]):
        if i < kernel_size:
            # print(sum(attn_weight[:i+1]))
            aa_attn[i] = sum(attn[:i + 1]) / (i + 1)
        else:
            aa_attn[i] = sum(attn[i - kernel_size:i + 1]) / kernel_size
    return aa_attn

