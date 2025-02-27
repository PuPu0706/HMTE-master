from functools import wraps, partial
from collections import namedtuple
from packaging import version
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat, pack, unpack
from beartype.typing import Optional, List
from mamba_ssm.modules.mamba2 import Mamba2
from .SML import SelectiveMemoryLayer


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner
def and_reduce(arr: List[torch.Tensor]):
    if len(arr) == 0:
        return None
    head, *rest = arr
    for t in rest:
        head = head & t
    return head
def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d
def pack_one(t, pattern):
    return pack([t], pattern)
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]
def l2norm(t):
    return F.normalize(t, dim=-1)
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(t, pos, scale=1.):
    scale = default(scale, 1.)

    seq_len = t.shape[-2]

    assert pos.shape[-2] >= seq_len

    pos = pos[-seq_len:]

    if isinstance(scale, torch.Tensor):
        assert scale.shape[-2] >= seq_len
        scale = scale[-seq_len:]

    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)



Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
print_once = once(print)



class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class Attend(nn.Module):
    def __init__(
            self,
            causal=False,
            use_flash_attn=False
    ):
        super().__init__()
        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash_attn = use_flash_attn
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse(
            '2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash_attn:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h=q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h=q.shape[1])

        masks = []

        if self.causal:
            i, j = q_len, k_len
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=q.device).triu(j - i + 1)
            masks.append(~causal_mask)

        if exists(mask):
            if mask.ndim != 2:
                mask = repeat(mask, 'w ... -> (b w) ...', b=q.shape[0] // mask.shape[0])

            masks.append(mask)

        attn_mask = and_reduce(masks)

        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask
            )

        return out

    def forward(self, q, k, v, pos_query, pos_key, pos_value, mask=None, use_flash_attn=None):
        use_flash_attn = default(use_flash_attn, self.use_flash_attn)

        b, n, device = q.shape[0], q.shape[-2], q.device

        q, ps = pack_one(q, '* h n d')
        k, _ = pack_one(k, '* n d')
        v, _ = pack_one(v, '* n d')

        if use_flash_attn:
            out = self.flash_attn(q, k, v, mask=mask)
            return unpack_one(out, ps, '* h n d')

        scale = q.shape[-1] ** -0.5

        k_einsum = 'b j d' if k.ndim == 3 else 'b h j d'
        v_einsum = 'b j d' if v.ndim == 3 else 'b h j d'

        sim = einsum(f"b h i d, {k_einsum} -> b h i j", q, k) * scale
        sim_q_poskey = einsum(f"b h i d, b j d -> b h i j", q[:, :, 1:, :], pos_key) * scale
        sim_posquery_k = einsum(f"b h d, b j d -> b h j", pos_query, k[:, 1:, :]) * scale
        sim_posquery_poskey = einsum(f"b h d, b j d -> b h j", pos_query, pos_key) * scale
        sim[:, :, 1:, 1:] = sim[:, :, 1:, 1:] + sim_q_poskey + sim_posquery_k.unsqueeze(
            1) + sim_posquery_poskey.unsqueeze(1)

        if exists(mask):
            if mask.ndim != 2:
                mask = repeat(mask, 'w ... -> (b w) ...', b=b)

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=q.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)

        out = einsum(f"b h i j, {v_einsum} -> b h i d", attn, v)
        out_pos = einsum(f"b h i j, b j d -> b h i d", attn[:, :, 1:, 1:], pos_value)
        out[:, :, 1:, :] = out[:, :, 1:, :] + out_pos
        return unpack_one(out, ps, '* h n d')

class Attention(nn.Module):
    def __init__(
            self,
            dim_head,
            causal=False,
            qk_rmsnorm=False,
            qk_rmsnorm_scale=8,
            use_flash_attn=False
    ):
        super().__init__()
        self.causal = causal

        self.qk_rmsnorm = qk_rmsnorm
        self.qk_rmsnorm_scale = qk_rmsnorm_scale

        self.attend = Attend(causal=causal, use_flash_attn=use_flash_attn)

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

    def forward(
            self,
            q, k, v,
            pos_query, pos_key, pos_value,
            mask=None,
            rotary_pos_emb=None,
            xpos_scale=None
    ):

        scale = q.shape[-1] ** -0.5

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            scale = self.qk_rmsnorm_scale

        if self.qk_rmsnorm:
            q = q * self.q_scale
            k = k * self.k_scale

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb, xpos_scale)
            k = apply_rotary_pos_emb(k, rotary_pos_emb, xpos_scale ** -1)
            pos_query = apply_rotary_pos_emb(pos_query, rotary_pos_emb, xpos_scale)
            pos_key = apply_rotary_pos_emb(pos_key, rotary_pos_emb, xpos_scale ** -1)

        out = self.attend(q, k, v, pos_query, pos_key, pos_value, mask=mask)

        return out

class StateContainer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            num_state_vectors,
            dim_head=64,
            heads=8,
            qk_rmsnorm=False,
            qk_rmsnorm_scale=8,
            use_flash_attn=False
    ):
        super().__init__()
        assert num_state_vectors > 0
        self.heads = heads
        inner_dim = dim_head * heads

        self.state_norm = LayerNorm(dim)

        self.q_to_state = nn.Linear(dim, inner_dim, bias=False)
        self.q_from_state = nn.Linear(dim, inner_dim, bias=False)

        self.state_to_q = nn.Linear(dim, inner_dim, bias=False)
        self.state_to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.arity = num_state_vectors
        self.next_read_state = nn.Parameter(torch.randn(self.arity, self.arity, dim))
        torch.nn.init.normal_(self.next_read_state, 0, .1)
        self.state_pos_ids = nn.Parameter(torch.randn(self.arity, dim))

        torch.nn.init.normal_(self.state_pos_ids, 0, 1)

        self.to_state_out = SelectiveMemoryLayer(Mamba2(d_model=dim, d_state=16, d_conv=4, expand=2, headdim=64), dim)
        self.to_state_cross_attn = Attention(dim_head, qk_rmsnorm=qk_rmsnorm, qk_rmsnorm_scale=qk_rmsnorm_scale,
                                             use_flash_attn=use_flash_attn)

        self.state_self_attn = Attention(dim_head, qk_rmsnorm=qk_rmsnorm, qk_rmsnorm_scale=qk_rmsnorm_scale,
                                         use_flash_attn=use_flash_attn)
        self.from_state_cross_attn = Attention(dim_head, qk_rmsnorm=qk_rmsnorm, qk_rmsnorm_scale=qk_rmsnorm_scale,
                                               use_flash_attn=use_flash_attn)

        self.state_out_to_gate = SelectiveMemoryLayer(Mamba2(d_model=dim, d_state=16, d_conv=4, expand=2, headdim=64),
                                                      dim)
        self.learned_ema_beta = nn.Parameter(torch.randn(dim))
        torch.nn.init.normal_(self.learned_ema_beta, 0, .1)

        self.cache = None

    def read(self, x, miss):
        assert exists(self.next_read_state), 'states to be read must be set with .set_next_read_state'

        states = self.next_read_state[miss - 1]

        normed_states = self.state_norm(states)

        normed_states = normed_states + self.state_pos_ids

        q_to_state = self.q_to_state(x)
        q_to_state = rearrange(q_to_state, '... n (h d) -> ... h n d', h=self.heads)

        state_k, state_v = self.state_to_kv(normed_states).chunk(2, dim=-1)

        to_state_out = self.to_state_cross_attn(q_to_state, state_k, state_v)

        to_state_out = rearrange(to_state_out, 'b h n d -> b n (h d)')

        return to_state_out

    def write(
            self,
            *,
            memories
    ):
        assert exists(self.cache)

        k, v = memories
        batch = k.shape[0]

        states, normed_states, state_k, state_v = self.cache

        self.cache = None

        q_from_state = self.q_from_state(normed_states)
        q_from_state = rearrange(q_from_state, '... n (h d) -> ... h n d', h=self.heads)

        state_q = self.state_to_q(normed_states)
        state_q_einsum = 'n (h d)' if state_q.ndim == 2 else 'b n (h d)'
        state_q = repeat(state_q, f'{state_q_einsum} -> b h n d', h=self.heads, b=batch)

        if q_from_state.ndim == 3:
            q_from_state = repeat(q_from_state, '... -> b ...', b=batch)

        state_out = self.state_self_attn(state_q, state_k, state_v)

        from_state_out = self.from_state_cross_attn(q_from_state, k, v)

        state_out = torch.cat((state_out, from_state_out), dim=-1)
        state_out = rearrange(state_out, 'b h n d -> b n (h d)')

        state_out = self.to_state_out(state_out)

        z = self.state_out_to_gate(state_out)
        learned_ema_decay = self.learned_ema_beta.sigmoid()

        return learned_ema_decay * z + (1 - learned_ema_decay) * states

    def forward(self, x):
        raise NotImplementedError


class PositionAwareSelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            block_width,
            dim_head=64,
            heads=8,
            qk_rmsnorm=False,
            qk_rmsnorm_scale=8,
            use_flash_attn=False,
            num_state_vectors=0,
            num_external_state_reads=0,
            state_read_before_write=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.attn = Attention(dim_head, qk_rmsnorm=qk_rmsnorm, qk_rmsnorm_scale=qk_rmsnorm_scale,
                              use_flash_attn=use_flash_attn)

        self.block_width = block_width
        self.is_recurrent_layer = num_state_vectors > 0

        num_state_reads = int(self.is_recurrent_layer and state_read_before_write) + num_external_state_reads

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.dim = dim
        if not self.is_recurrent_layer:
            return

        self.state_read_before_write = state_read_before_write

        self.state_container = StateContainer(
            dim,
            dim_head=dim_head,
            heads=heads,
            num_state_vectors=num_state_vectors,
            qk_rmsnorm=qk_rmsnorm,
            qk_rmsnorm_scale=qk_rmsnorm_scale,
            use_flash_attn=use_flash_attn
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
            self,
            x,
            pos_query=None,
            pos_key=None,
            pos_value=None,
            rotary_pos_emb=None,
            xpos_scale=None,
            attn_mask=None,
            xl_memories: Optional[torch.Tensor] = None,
            miss_ent_domain=None,
            read_from_state_containers: List[StateContainer] = []
    ):
        batch, seq_len, _, width, device = *x.shape, self.block_width, self.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        split_head = partial(rearrange, pattern='b n (h d) -> b h n d', h=self.heads)
        q = split_head(q)

        memories = torch.stack((k, v))

        mem_len = 0

        if exists(xl_memories):
            mem_len = xl_memories.shape[-2]
            past_k, past_v = xl_memories
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)

        if exists(attn_mask):
            attn_mask = attn_mask[:seq_len, :seq_len]
            attn_mask = F.pad(attn_mask, (mem_len, 0), value=True)

        out = self.attn(
            q, k, v,
            pos_query, pos_key, pos_value,
            rotary_pos_emb=rotary_pos_emb,
            xpos_scale=xpos_scale,
            mask=None
        )

        out = rearrange(out, 'b h n d -> b n (h d)')

        if not self.is_recurrent_layer and len(read_from_state_containers) == 0:
            return self.to_out(out), memories, None

        if self.is_recurrent_layer and self.state_read_before_write:
            read_from_state_containers = [self.state_container, *read_from_state_containers]

        for read_state_container in read_from_state_containers:
            to_state_out = read_state_container.read(x, miss_ent_domain)

            out = out + to_state_out

        new_states = None

        return self.to_out(out), memories, new_states