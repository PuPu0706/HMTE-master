import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from itertools import zip_longest
from beartype.typing import List

def exists(val):
    return val is not None
def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)
def is_empty(t: torch.Tensor):
    return t.numel() == 0
def compact(arr):
    return [*filter(exists, arr)]
def safe_cat(*args, dim=1):
    args = compact(args)

    if len(args) == 0:
        return None

    return torch.cat(args, dim=dim)



class AM_Layer(nn.Module):
    def __init__(self, self_attention, mamba, d_model, dropout):
        super(AM_Layer, self).__init__()
        self.self_attention = self_attention
        self.mamba = mamba
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask=None):
        out, attn = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(out)
        x = self.norm1(x)

        x = x + self.mamba(x)
        x = self.norm2(x)

        return x, attn

class Mamba_Layer(nn.Module):
    def __init__(self, mamba, d_model):
        super(Mamba_Layer, self).__init__()
        self.mamba = mamba
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.mamba(x)
        x = self.norm(x)

        return x

class SelectiveMemoryLayer(nn.Module):
    def __init__(self, sml, d_model):
        super(SelectiveMemoryLayer, self).__init__()
        self.sml = sml
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sml(x)
        x = self.norm(x)
        return x

class MemoryManager(nn.Module):
    def __init__(
            self,
            dim,
            *,
            layers=1,
            mem_lengths=512,
            compress_factors=1
    ):
        super().__init__()
        mem_lengths = cast_tuple(mem_lengths)
        compress_factors = cast_tuple(compress_factors)

        assert all([mem_length > 0 for mem_length in mem_lengths])
        assert len(mem_lengths) == len(compress_factors)
        assert layers >= 1

        self.mem_lengths = mem_lengths
        self.compress_factors = compress_factors

        self.layers = nn.ModuleList([])

        for _ in range(layers):
            compress_fns = nn.ModuleList([])

            for compress_factor in compress_factors:
                compress_fn = nn.Identity()
                if compress_factor > 1:
                    compress_fn = nn.Sequential(
                        Rearrange('b n d -> b d n'),
                        nn.Conv1d(
                            dim * 2,
                            dim * 2,
                            compress_factor,
                            stride=compress_factor,
                            groups=2
                        ),
                        Rearrange('b d n -> b n d'),
                    )

                compress_fns.append(compress_fn)

            self.layers.append(compress_fns)

    def forward(
            self,
            past_memories: List[torch.Tensor],
            new_memories: List[torch.Tensor]
    ):
        next_memories = []

        for past_memory, new_memory, compress_fns in zip_longest(past_memories, new_memories, self.layers):

            if not (exists(past_memory) or exists(new_memory)):
                next_memories.append(None)
                continue

            next_memory = None

            for mem_length, compress_factor, compress_fn in zip(self.mem_lengths, self.compress_factors, compress_fns):

                current_memory = None
                if exists(past_memory):
                    past_memory, current_memory = past_memory[..., :-mem_length, :], past_memory[..., -mem_length:, :]

                if (not is_empty(new_memory)) and compress_factor > 1:

                    new_mem_length = new_memory.shape[-2]

                    curtailed_length = (new_mem_length // compress_factor) * compress_factor

                    curtailed_slice = slice(-curtailed_length, None) if curtailed_length > 0 else slice(0, 0)
                    new_memory = new_memory[..., curtailed_slice, :]

                    if new_memory.shape[-2] > 0:
                        new_memory = rearrange(new_memory, 'm b n d -> b n (m d)')
                        new_memory = compress_fn(new_memory)
                        new_memory = rearrange(new_memory, 'b n (m d) -> m b n d', m=2)

                current_memory = safe_cat(current_memory, new_memory, dim=-2)

                new_memory, current_memory = current_memory[..., :-mem_length, :], current_memory[..., -mem_length:, :]

                next_memory = safe_cat(current_memory, next_memory, dim=-2)

            next_memories.append(next_memory)

        return next_memories
