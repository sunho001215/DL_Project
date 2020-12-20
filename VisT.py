"""
https://github.com/tahmid0007/VisualTransformers/blob/main/ResViT.py
Based on the code above
"""

import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x

class VisTGen(nn.Module):
    def __init__(self, batch_size=512 ,ngf=64, nz=100, nc=3, num_classes=3, dim = 128, num_tokens = 8, mlp_dim = 256, heads = 8, depth = 6, emb_dropout = 0.1, dropout= 0.1):
        super(VisTGen, self).__init__()

        #self.batch_size = batch_size
        self.tconv1 = nn.ConvTranspose2d(nz, ngf*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)
        self.tconv2 = nn.ConvTranspose2d(ngf*8, ngf*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)
        self.tconv3 = nn.ConvTranspose2d(ngf*4, ngf*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)
        self.tconv4 = nn.ConvTranspose2d(ngf*2, ngf,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.tconv5 = nn.ConvTranspose2d(ngf, nc,
            4, 2, 1, bias=False)
        #self.bn5 = nn.BatchNorm2d(nc)

        self.L = num_tokens
        self.cT = dim
        
        # Tokenization
        #self.token_wA = nn.Parameter(torch.empty(batch_size,self.L, 3),requires_grad = True) #Tokenization parameters
        self.token_wA_pre = nn.Parameter(torch.empty(self.L, 3),requires_grad = True)
        torch.nn.init.xavier_normal_(self.token_wA_pre)
        #self.token_wV = nn.Parameter(torch.empty(batch_size,3,self.cT),requires_grad = True) #Tokenization parameters
        self.token_wV_pre = nn.Parameter(torch.empty(3,self.cT),requires_grad = True)
        torch.nn.init.xavier_normal_(self.token_wV_pre)        
             
        
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding) # initialized based on the paper

        #self.patch_conv= nn.Conv2d(64,dim, self.patch_size, stride = self.patch_size) 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        #self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        #self.Q_TtoX = nn.Parameter(torch.empty(batch_size, 3, 3), requires_grad = True)
        self.Q_TtoX_pre = nn.Parameter(torch.empty(3, 3), requires_grad = True)
        torch.nn.init.xavier_normal_(self.Q_TtoX_pre)
        #self.K_TtoX = nn.Parameter(torch.empty(batch_size, dim, 3), requires_grad = True)
        self.K_TtoX_pre = nn.Parameter(torch.empty(dim, 3), requires_grad = True)
        torch.nn.init.xavier_normal_(self.K_TtoX_pre)
        #self.V_TtoX = nn.Parameter(torch.empty(batch_size, dim, 3), requires_grad = True)
        self.V_TtoX_pre = nn.Parameter(torch.empty(dim, 3), requires_grad = True)
        torch.nn.init.xavier_normal_(self.V_TtoX_pre)

        #self.nn1 = nn.Linear(dim, num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        #torch.nn.init.xavier_uniform_(self.nn1.weight)
        #torch.nn.init.normal_(self.nn1.bias, std = 1e-6)


        
    def forward(self, img, mask=None):
        batch_size = img.shape[0]
        x = F.relu(self.bn1(self.tconv1(img)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x_in = self.tconv5(x)
        #print(x_in.shape)
        
        x = rearrange(x_in, 'b c h w -> b (h w) c') # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP

        #Tokenization 
        token_wA = torch.unsqueeze(self.token_wA_pre,0).repeat(batch_size, 1, 1)
        wa = rearrange(token_wA, 'b h w -> b w h') #Transpose
        A= torch.einsum('bij,bjk->bik', x, wa) 
        A = rearrange(A, 'b h w -> b w h') #Transpose
        A = A.softmax(dim=-1)

        token_wV = torch.unsqueeze(self.token_wV_pre,0).repeat(batch_size, 1, 1)
        VV= torch.einsum('bij,bjk->bik', x, token_wV)       
        T = torch.einsum('bij,bjk->bik', A, VV)  
        #print(T.shape)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        #x = self.dropout(x)
        #print(x.shape)
        Tout = self.transformer(x, mask) #main game
        #print(Tout.shape)

        Q_TtoX = torch.unsqueeze(self.Q_TtoX_pre,0).repeat(batch_size, 1, 1)
        K_TtoX = torch.unsqueeze(self.K_TtoX_pre,0).repeat(batch_size, 1, 1)
        V_TtoX = torch.unsqueeze(self.V_TtoX_pre,0).repeat(batch_size, 1, 1)
        x = rearrange(x_in, 'b c h w -> b (h w) c')
        M_1 = torch.einsum('bij,bjk->bik', x, Q_TtoX)
        M_2 = torch.einsum('bij,bjk->bik', Tout, K_TtoX)
        M_2_T = rearrange(M_2, 'b h w -> b w h')
        M_3 = torch.einsum('bij,bjk->bik', M_1, M_2_T)
        M_softmax = M_3.softmax(dim=-1)

        M_4 = torch.einsum('bij,bjk->bik', Tout, V_TtoX)
        M_fin = torch.einsum('bij,bjk->bik', M_softmax, M_4)

        out = rearrange(M_fin, 'b s c -> b c s')
        out = torch.tanh(torch.reshape(out, (batch_size, 3, 64, 64)) + x_in)
        #x = self.to_cls_token(x[:, 0])       
        #x = self.nn1(x)

        return torch.tanh(x_in), out

class VisTDis(nn.Module):
    def __init__(self, batch_size=512 ,ndf=64, nz=100, nc=3, num_classes=1, dim = 128, num_tokens = 8, mlp_dim = 256, heads = 8, depth = 6, emb_dropout = 0.1, dropout= 0.1):
        super(VisTDis, self).__init__()

        self.cv1 = nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False) # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1 ) # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        self.cv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1) # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.cv4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False) # (256, 8, 8) -> (512, 4, 4)
        self.bn4 = nn.BatchNorm2d(ndf* 8)

        self.L = num_tokens
        self.cT = dim
         # Tokenization
        #self.token_wA = nn.Parameter(torch.empty(batch_size,self.L, 3),requires_grad = True) #Tokenization parameters
        self.token_wA_pre = nn.Parameter(torch.empty(self.L, 256),requires_grad = True)
        torch.nn.init.xavier_uniform_(self.token_wA_pre)
        #self.token_wV = nn.Parameter(torch.empty(batch_size,3,self.cT),requires_grad = True) #Tokenization parameters
        self.token_wV_pre = nn.Parameter(torch.empty(256,self.cT),requires_grad = True)
        torch.nn.init.xavier_uniform_(self.token_wV_pre)         
             
        
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std = .02) # initialized based on the paper

        #self.patch_conv= nn.Conv2d(64,dim, self.patch_size, stride = self.patch_size) 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        #self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)

    def forward(self, img, mask = None):
        batch_size = img.shape[0]

        x = F.leaky_relu(self.cv1(img))
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        #x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)

        x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP

        #Tokenization 
        token_wA = torch.unsqueeze(self.token_wA_pre,0).repeat(batch_size, 1, 1)
        token_wV = torch.unsqueeze(self.token_wV_pre,0).repeat(batch_size, 1, 1)
        wa = rearrange(token_wA, 'b h w -> b w h') #Transpose
        A= torch.einsum('bij,bjk->bik', x, wa) 
        A = rearrange(A, 'b h w -> b w h') #Transpose
        A = A.softmax(dim=-1)

        VV= torch.einsum('bij,bjk->bik', x, token_wV)       
        T = torch.einsum('bij,bjk->bik', A, VV)  
        #print(T.size())

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        #x = self.dropout(x)
        x = self.transformer(x, mask) #main game
        x = self.to_cls_token(x[:, 0])       
        x = self.nn1(x)

        x = torch.sigmoid(x)

        return x
