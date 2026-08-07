import torch
import torch.nn as nn
import torch.nn.functional as FF
import numpy as np
import math
from typing import Tuple
from utils.train_utils import gft_patches, igft_patches


class MLP(nn.Module):
    def __init__(self, dim_in, dropout, scale_ratio=4, dim_out=None, num_layers=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in

        self.residual_projection = None
        if dim_in != dim_out:
            self.residual_projection = nn.Linear(dim_in, dim_out)

        scaled_dim = max(1, int(dim_in * scale_ratio))

        layers = []
        if num_layers is None or num_layers == 1:
            layers.append(nn.Linear(dim_in, scaled_dim))
            layers.append(nn.LayerNorm(scaled_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(scaled_dim, dim_out))
        else:
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(dim_in, scaled_dim))
                    layers.append(nn.LayerNorm(scaled_dim))
                elif i == num_layers - 1:
                    layers.append(nn.Linear(scaled_dim, dim_out))
                else:
                    layers.append(nn.Linear(scaled_dim, scaled_dim))
                    layers.append(nn.LayerNorm(scaled_dim))

                if i != num_layers - 1:  # Don't add GELU after the last layer
                    layers.append(nn.GELU())

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.dropout(x)

class upScaleMLP(nn.Module):
    def __init__(self,d_model,d_output,hidden_dim):
        super(upScaleMLP, self).__init__()
        self.d_model = d_model
        self.d_output = d_output
        self.hidden_dim = hidden_dim
        self.layer1 = nn.Linear(d_model,hidden_dim,bias=False)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim,d_output)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

class downScaleMLP(nn.Module):
    def __init__(self,d_input,d_model,hidden_dim):
        super(downScaleMLP, self).__init__()
        self.d_model = d_model
        self.d_input = d_input
        self.layer1 = nn.Linear(d_input,hidden_dim,bias=False)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim,d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input, cond=None):
        return FF.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# Not masked for encoder part
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B,T,C = x.shape
        # B,T,C
        k = self.k(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        q = self.q(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = self.v(x).view(B,T,self.n_heads,self.head_dim).transpose(1,2)

        attn =(q @ k.transpose(-2,-1) ) * (self.head_dim)**-0.5
        attn = FF.softmax(attn,dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        out = self.projection(out)

        return out

class EncoderBlock(nn.Module):

    def __init__(self, n_heads, max_len, embed_dim, src_len, dropout):
        super().__init__()

        self.ln_exp1_1 = LayerNorm(embed_dim, bias=False)
        self.ln_exp1_2 = LayerNorm(embed_dim, bias=False)

        self.attn_1 = MultiHeadAttention(n_heads, embed_dim, dropout)

        self.mlp_1 = MLP(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn_1(self.ln_exp1_1(x))
        output = x + self.mlp_1(self.ln_exp1_2(x))
        return output
    

# Taken from the "Score-Based Generative Modeling Through Stochastic Differential Equations"
# https://github.com/yang-song/score_sde_pytorch/tree/main
class GaussianFourierProjection(nn.Module):
    def __init__(self, input_dim, half_dim=256, scale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.half_dim = half_dim
        self.W = nn.Parameter(torch.randn(input_dim, half_dim) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x @ self.W * 2 * np.pi  # [B,T,input_dim] @ [input_dim,half_dim] -> [B,T,half_dim]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # [B,T,2*half_dim]

#-------------------------------------------------------------------------------------------------
# Masked Multi-Head Attention with RoPE
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, max_len, src_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim * n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim * n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim * n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # Precompute freqs_cis for RoPE
        freqs_cis = precompute_freqs_cis(self.head_dim, max_len)
        self.register_buffer('freqs_cis', freqs_cis)

        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len), diagonal=src_len).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.shape  # B,T,C
        
        # Project to q, k, v
        q = self.q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim)

        # Apply rotary positional embeddings
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim)**-0.5
        attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        attn = FF.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        out = self.projection(out)

        return out
#-------------------------------------------------------------------------------------------------
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, max_len, src_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim*n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

        self.register_buffer('tril',torch.tril(torch.ones(max_len,max_len), diagonal=src_len).view(1,1,max_len,max_len))

    def forward(self, x_1, x_2): # x_1 comes from decoder, x_2 comes from encoder
        """X1 And X2 have different shapes restructure q, k, v according to their own shapes"""
        B_tgt,T_tgt,C_tgt = x_1.shape
        B_src,T_src,C_src = x_2.shape

        q = self.q(x_1).view(B_tgt,T_tgt,self.n_heads,self.head_dim).transpose(1,2)

        v = self.v(x_2).view(B_src,T_src,self.n_heads,self.head_dim).transpose(1,2)
        k = self.k(x_2).view(B_src,T_src,self.n_heads,self.head_dim).transpose(1,2)

        attn = (q @ k.transpose(-2,-1) ) * (self.head_dim)**-0.5 
        
        attn = FF.softmax(attn,dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B_tgt, T_tgt, C_tgt)

        out = self.projection(out)

        return out

#-------------------------------------------------------------------------------------------------
# Masked Multi-Head Cross Attention with RoPE
class MaskedMultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, max_len, src_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.head_dim = embed_dim // n_heads
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, self.head_dim * n_heads)
        self.q = nn.Linear(embed_dim, self.head_dim * n_heads)
        self.v = nn.Linear(embed_dim, self.head_dim * n_heads)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # Precompute freqs_cis for RoPE
        freqs_cis = precompute_freqs_cis(self.head_dim, max_len)
        self.register_buffer('freqs_cis', freqs_cis)

        # Registered as buffer so it's saved and loaded with the model
        self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len), diagonal=src_len).view(1, 1, max_len, max_len))

    def forward(self, x_1, x_2):  # Assume both x_1, and x_2 have the same shape
        B, T, C = x_1.shape  # B,T,C

        # Project inputs to q, k, v
        q = self.q(x_1).view(B, T, self.n_heads, self.head_dim)  # [B, n_H, T, H_dim]
        k = self.k(x_2).view(B, T, self.n_heads, self.head_dim)  # [B, n_H, T, H_dim]
        v = self.v(x_2).view(B, T, self.n_heads, self.head_dim)  # [B, n_H, T, H_dim]

        # Apply rotary positional embeddings
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim)**-0.5  # [B, n_H, T, T]
        attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        attn = FF.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, n_H, T, T] * [B, n_H, T, H_dim] = [B, n_H, T, H_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, N_H*H_dim] = [B, T, C]

        # Final projection
        out = self.projection(out)

        return out
    
#-------------------------------------------------------------------------------------------------
# RoPE
# Taken from the llama3 repo: https://github.com/meta-llama/llama3/blob/main/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

#-------------------------------------------------------------------------------------------------
# Adaptive Layer Normalization
# The AdaLN is also available in https://github.com/huggingface/diffusers/blob/v0.30.3/src/diffusers/models/normalization.py#L31
# This class is similar to diffusers version with an additional linear layer and it takes the condition as input rather than time (Like diffusion models)
class AdaLN(nn.Module):
    def __init__(self, embed_dim, cond_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.cond_dim = cond_dim
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 2 * embed_dim),
            nn.SiLU(),
            nn.Linear(2 * embed_dim, 2 * embed_dim)
        )

    def forward(self, x, cond):
        cond = self.cond_mlp(cond)
        weight, bias = cond.chunk(2, dim=-1)
        weight = weight + 1
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (var + 1e-5).sqrt()
        return normalized * (self.weight + weight) + (self.bias + bias)


#-------------------------------------------------------------------------------------------------
# Clasical Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # Ensure div_term matches the size
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionalEncodingLearnable(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super(PositionalEncodingLearnable, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        seq_len = x.size(1)
        pos_embeddings = self.pos_embedding[:, :seq_len, :]
        x = x + pos_embeddings
        return self.dropout(x)

class SpectralBlock(nn.Module):
    def __init__(self, latent_dim, U_patch: torch.Tensor):
        super().__init__()
        # self.ln1 = LayerNorm(latent_dim)
        # self.ln2 = LayerNorm(latent_dim)
        self.U_patch = U_patch
        K = self.U_patch.shape[-1]
        self.W = nn.Parameter(torch.randn(U_patch.shape[0], latent_dim, K, K) * 0.02, requires_grad=True)
        self.b = nn.Parameter(torch.zeros(U_patch.shape[0], latent_dim, K), requires_grad=True)
        self.activation = nn.GELU()

    def forward(self, z):
        self.U_patch = self.U_patch.to(z.device)  # Ensure U_patch is on the same device
        z_res = z
        # z = self.ln1(z)
        z = gft_patches(z, self.U_patch)
        z = torch.einsum("bpfk,pfkm->bpfm", z, self.W) + self.b
        z = igft_patches(z, self.U_patch)
        # z = self.ln2(z)
        return z_res + self.activation(z)  # Residual connection
    
class SpectralAttentionBlock(nn.Module):
    def __init__(self, latent_dim, U_patch: torch.Tensor, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.U_patch = U_patch
        self.K_orig = self.U_patch.shape[-1]  # Original number of spectral coefficients
        self.num_heads = num_heads

        # === 1. Calculate the required padding ===
        if self.K_orig % self.num_heads != 0:
            # How many values are missing to make it a multiple of num_heads
            self.padding = self.num_heads - (self.K_orig % self.num_heads)
            K_padded = self.K_orig + self.padding
        else:
            self.padding = 0
            K_padded = self.K_orig
        
        # === 2. Initialize layers with the PADDED dimension ===
        self.ln1 = nn.LayerNorm(K_padded)
        self.attention = nn.MultiheadAttention(
            embed_dim=K_padded, # <-- MODIFIED
            num_heads=num_heads,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(K_padded)
        self.mlp = nn.Sequential(
            nn.Linear(K_padded, int(K_padded * mlp_ratio)), # <-- MODIFIED
            nn.GELU(),
            nn.Linear(int(K_padded * mlp_ratio), K_padded), # <-- MODIFIED
        )

    def forward(self, z):
        # z has shape (b, p, f, K_orig)
        self.U_patch = self.U_patch.to(z.device)
        z_res = z

        # === Graph Fourier Transform ===
        z_spectral = gft_patches(z, self.U_patch) # Shape: (b, p, f, K_orig)

        # === 3. Pad the spectral dimension ===
        # The pad tuple (0, self.padding) applies to the last dimension.
        # It adds 0 elements to the left and self.padding elements to the right.
        if self.padding > 0:
            # Shape becomes (b, p, f, K_padded)
            z_padded = FF.pad(z_spectral, (0, self.padding), "constant", 0)
        else:
            z_padded = z_spectral

        # === Prepare for Attention ===
        # The user's original logic had a bug in reshaping and comments.
        # MultiheadAttention expects (batch, seq, embed_dim).
        # To perform attention over spectral coefficients (the sequence), we need to permute.
        b, p, f, K_pad = z_padded.shape
        # Permute from (b, p, f, K_pad) to (b, p, K_pad, f)
        z_permuted = z_padded.permute(0, 1, 3, 2) 
        # Reshape for attention: (b*p, K_pad, f)
        z_reshaped = z_permuted.reshape(b * p, K_pad, f)

        # === Apply Transformer Block (Attention + FFN) ===
        # NOTE: The LayerNorm and MLP now expect the last dimension to be `f`, not `K_padded`.
        # I am correcting the user's original implementation to match the standard transformer logic
        # where attention is applied over a sequence, and the MLP is applied to the features of each item.
        # If the user *truly* intended to apply attention across features `f`, their original reshape
        # `z_spectral.reshape(b * p, f, K)` was correct, and my code above should be adapted.
        # Assuming the standard case (attention over the K dimension):
        
        # This part of the code needs to be adapted based on what dimension attention is over.
        # Let's assume the user's original intent was correct: attention over `f` (latent_dim).
        b, p, f, K_pad = z_padded.shape
        z_reshaped = z_padded.reshape(b * p, f, K_pad) # (batch, seq=f, embed_dim=K_pad)

        # 1. Attention Block
        norm_z = self.ln1(z_reshaped)
        attn_output, _ = self.attention(norm_z, norm_z, norm_z)
        z_reshaped = z_reshaped + attn_output

        # 2. FFN Block
        norm_z = self.ln2(z_reshaped)
        mlp_output = self.mlp(norm_z)
        z_reshaped = z_reshaped + mlp_output
        
        # === 4. Un-pad by slicing ===
        # Reshape back to 4D tensor: (b, p, f, K_padded)
        z_padded_out = z_reshaped.reshape(b, p, f, K_pad)
        
        # Slice the last dimension to remove the padding
        if self.padding > 0:
            # Shape becomes (b, p, f, K_orig)
            z_spectral_out = z_padded_out[..., :self.K_orig]
        else:
            z_spectral_out = z_padded_out

        # === Inverse Transform ===
        z_out = igft_patches(z_spectral_out, self.U_patch)

        return z_res + z_out

# class FNOBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels):
#         super().__init__()
#         self.ln1 = LayerNorm(in_channels)
#         self.ln2 = LayerNorm(in_channels)
#         self.W = nn.Parameter(torch.randn(hidden_channels, in_channels, out_channels) * 0.02, requires_grad=True)
#         self.activation = nn.GELU()

#     def forward(self, z):
#         self.U_patch = self.U_patch.to(z.device)  # Ensure U_patch is on the same device
#         z_res = z
#         z = self.ln1(z)
#         z = gft_patches(z, self.U_patch)
#         z = self.Linear(z)
#         z = igft_patches(z, self.U_patch)
#         z = self.ln2(z)
#         return z_res + self.activation(z)  # Residual connection