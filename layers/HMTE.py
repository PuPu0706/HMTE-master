import math
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from beartype import beartype
from beartype.typing import Optional, List, Tuple
from mamba_ssm.modules.mamba2 import Mamba2

from .PASA import PositionAwareSelfAttention
from .SML import SelectiveMemoryLayer, MemoryManager


def exists(val):
    return val is not None
def all_unique(arr):
    return len(arr) == len(set(arr))
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner
def divisible_by(numer, denom):
    return (numer % denom) == 0
def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))
def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)
def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs
def default(val, d):
    return val if exists(val) else d



class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            width,
            scale_base=512,
            theta=10000
    ):
        super().__init__()
        self.width = width

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent=False)

        self.register_buffer('cached_freqs', None, persistent=False)
        self.register_buffer('cached_scales', None, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self):
        device, seq_len = self.device, self.width

        if exists(self.cached_freqs):
            cached_seq_len = self.cached_freqs.shape[-2]
            if cached_seq_len >= seq_len:
                return self.cached_freqs[:seq_len], self.cached_scales[:seq_len]

        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim=-1)

        self.register_buffer('cached_freqs', freqs, persistent=False)
        self.register_buffer('cached_scales', scale, persistent=False)
        return freqs, scale



@beartype
class Hmte_Model(nn.Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            all_layers_qk_rmsnorm=False,
            max_seq_len=1024,
            block_width=512,
            recurrent_layers: Optional[Tuple[int, ...]] = None,
            read_recurrent_layers: Optional[Tuple[int, ...]] = None,
            num_state_vectors=None,
            ignore_index=-100,
            use_flash_attn=False,
            use_compressed_mem=False,
            compressed_mem_factor=4
    ):
        super().__init__()
        num_state_vectors = default(num_state_vectors, block_width)
        recurrent_layers = default(recurrent_layers, (depth // 2,))

        assert all([0 < layer <= depth for layer in
                    recurrent_layers]), f'recurrent layers must range from 1 to the depth {depth}'
        assert all_unique(recurrent_layers), 'recurrent layers must be all unique. no duplicate layers'

        self.recurrent_layers = recurrent_layers
        read_recurrent_layers = default(read_recurrent_layers, recurrent_layers)

        assert all([read_layer <= write_layer for read_layer, write_layer in zip(read_recurrent_layers,
                                                                                 recurrent_layers)]), 'the recurrent read layer must be always less than or equal to the write layer'
        assert all([0 < layer <= depth for layer in read_recurrent_layers])
        assert len(read_recurrent_layers) == len(recurrent_layers)

        self.read_recurrent_layers = read_recurrent_layers
        self.rotary_pos_emb = RotaryEmbedding(dim=dim_head, width=(2 if not use_compressed_mem else 3) * block_width)
        self.layers = nn.ModuleList([])
        self.write_to_read_map = {write_layer: read_layer for write_layer, read_layer in
                                  zip(recurrent_layers, read_recurrent_layers)}
        self.read_state_router = defaultdict(list)
        for layer in range(1, depth + 1):
            is_recurrent_layer = layer in self.recurrent_layers

            layer_num_state_vectors = num_state_vectors if is_recurrent_layer else 0

            num_external_state_reads = sum([int(layer == read_layer) for read_layer in read_recurrent_layers])

            qk_rmsnorm = all_layers_qk_rmsnorm or is_recurrent_layer

            posAtt = PositionAwareSelfAttention(
                dim,
                block_width=block_width,
                dim_head=dim_head,
                heads=heads,
                qk_rmsnorm=qk_rmsnorm,
                num_state_vectors=layer_num_state_vectors,
                use_flash_attn=use_flash_attn,
                num_external_state_reads=num_external_state_reads,
                state_read_before_write=False,
            )

            SML = SelectiveMemoryLayer(Mamba2(d_model=dim, d_state=16, d_conv=4, expand=2, headdim=64), dim)
            if is_recurrent_layer:
                read_layer = self.write_to_read_map[layer]
                self.read_state_router[read_layer].append(posAtt.state_container)

            self.layers.append(nn.ModuleList([
                posAtt,
                SML
            ]))

        self.mem_manager = MemoryManager(
            dim=dim_head,
            layers=depth,
            mem_lengths=block_width if not use_compressed_mem else (block_width, block_width // 2),
            compress_factors=1 if not use_compressed_mem else (1, compressed_mem_factor)
        )

        self.max_seq_len = max_seq_len
        self.block_width = block_width

        assert divisible_by(max_seq_len, block_width)

        self.ignore_index = ignore_index
        self.register_buffer('cached_causal_attn_mask', None, persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_causal_attn_mask(self, width):
        if exists(self.cached_causal_attn_mask):
            cached_mask = self.cached_causal_attn_mask
            cached_width = cached_mask.shape[-2]
            padding = (width - cached_width) // 2
            j_slice = Ellipsis if padding == 0 else slice(padding, -padding)
            return cached_mask[:cached_width, j_slice]

        device = self.device
        causal_mask = torch.ones((width, width), device=device, dtype=torch.bool).triu(1)
        return ~causal_mask

    @torch.no_grad()
    @eval_decorator
    def generate(
            self,
            prime,
            length=None,
            xl_memories: List[torch.Tensor] = [],
            states: List[torch.Tensor] = [],
            temperature=1.,
            filter_thres=0.9,
            return_memories_and_states=False
    ):
        length = default(length, self.max_seq_len + 1)
        start_len = prime.shape[-1]

        assert start_len < self.max_seq_len
        assert length <= (self.max_seq_len + 1)
        assert start_len < length

        output = prime

        memories = []

        for _ in range(length - start_len):

            logits, next_memories, next_states = self.forward(
                output,
                xl_memories=xl_memories,
                states=states
            )

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres=filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature=temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            output = torch.cat((output, sampled), dim=-1)

            if divisible_by(output.shape[-1] - 1, self.max_seq_len):
                memories = next_memories
                states = next_states

        output = output[:, start_len:]

        if return_memories_and_states:
            return output, memories, states

        return output

    def forward(
            self,
            x,
            miss_ent_domain=None,
            pos_query=None,
            pos_key=None,
            pos_value=None,
    ):
        w = self.block_width
        attn_mask = self.get_causal_attn_mask(w)
        rotary_pos_emb, xpos_scale = self.rotary_pos_emb()
        batch, _, dim = x.shape

        out = torch.empty(batch, 0, dim, dtype=x.dtype, device=self.device)

        input_blocks = x.split(w, dim=-2)

        for input_block in input_blocks:
            for ind, (attn, ff) in enumerate(self.layers):
                layer = ind + 1
                attn_kwargs = dict(
                    rotary_pos_emb=rotary_pos_emb,
                    xpos_scale=xpos_scale,
                    attn_mask=attn_mask,
                    miss_ent_domain=miss_ent_domain,
                    read_from_state_containers=self.read_state_router[layer]
                )
                residual = input_block
                attn_branch_out, layer_xl_memories, layer_next_states = attn(input_block, pos_query, pos_key, pos_value,
                                                                             **attn_kwargs)
                input_block = attn_branch_out + residual
                input_block = ff(input_block)
            out = torch.cat((out, input_block), dim=-2)
        return out, None, None
