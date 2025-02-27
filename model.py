import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2
from layers.SML import Mamba_Layer
from layers.HMTE import Hmte_Model


class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        return

    def forward(self, pred1, tar1):
        loss = self.loss_fn(pred1, tar1)
        return loss


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            layer_num: int,
            ary=2,
    ) -> None:
        super().__init__()
        self.layer_num = layer_num
        self.d_model = d_model
        self.ary = ary + 2
        self.all_size = 32
        self.window_size = 4
        self.model = Hmte_Model(
            dim=d_model,
            depth=layer_num,
            dim_head=64,
            heads=8,
            max_seq_len=ary + 2,
            block_width=ary + 2,
            num_state_vectors=ary + 2,
            recurrent_layers=(),
            use_compressed_mem=False,
            compressed_mem_factor=4,
            use_flash_attn=False
        )

    def forward(self, x, miss_ent_domain, pos_query, pos_key, pos_value):
        out, mems1, states1 = self.model(x, miss_ent_domain, pos_query, pos_key, pos_value)
        return out


class HMTE(BaseClass):

    def __init__(self, n_ent, n_rel, emb_dim, max_arity, device, layer_num, ent_relnel, ary_list):
        super(HMTE, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.device = device
        self.emb_dim = emb_dim
        self.lmbda = 0.05
        self.max_arity = max_arity
        self.ent_embeddings = nn.Parameter(torch.Tensor(self.n_ent, self.emb_dim))
        self.propmt = nn.Parameter(torch.Tensor(1, self.emb_dim))
        self.neigh = ent_relnel
        self.pos_embeddings = nn.Embedding(self.max_arity, self.emb_dim)
        self.arylist = ary_list
        self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel, self.emb_dim))
        self.register_parameter('b', nn.Parameter(torch.zeros(n_ent)))
        self.preprocess = Mamba_Layer(Mamba2(d_model=emb_dim, d_state=16, d_conv=4, expand=2, headdim=64), emb_dim)
        self.decoder = nn.ModuleList(
            [MixerModel(d_model=emb_dim, layer_num=layer_num, ary=i) for i in range(len(ary_list) + 3)])
        self.norm = nn.LayerNorm(emb_dim)
        self.edge_query_embedding = nn.Embedding(self.max_arity, 64)
        self.edge_key_embedding = nn.Embedding(self.max_arity, 64)
        self.edge_value_embedding = nn.Embedding(self.max_arity, 64)

        nn.init.xavier_uniform_(self.edge_query_embedding.weight.data)
        nn.init.xavier_uniform_(self.edge_key_embedding.weight.data)
        nn.init.xavier_uniform_(self.edge_value_embedding.weight.data)

        nn.init.xavier_uniform_(self.ent_embeddings.data)
        nn.init.xavier_uniform_(self.rel_embeddings.data)
        nn.init.xavier_uniform_(self.pos_embeddings.weight.data)
        nn.init.xavier_uniform_(self.propmt.data)

    def get_neighbors(self, entity, max_entities=100, max_levels=3):
        neighbors = set()
        current_level_neighbors = {entity}
        for _ in range(1, max_levels + 1):
            next_level_neighbors = set()
            for e in current_level_neighbors:
                next_level_neighbors.update(self.neigh.get(e, []))
            neighbors.update(next_level_neighbors)
            current_level_neighbors = next_level_neighbors
            if len(neighbors) >= max_entities:
                break

        return list(neighbors)[:max_entities]

    def forward(self, rel_idx, ent_idx, miss_ent_domain):

        r = self.rel_embeddings[rel_idx].unsqueeze(1)
        ents = self.ent_embeddings[ent_idx]

        pos = [i for i in range(ent_idx.shape[1] + 1) if i + 1 != miss_ent_domain]
        pos = torch.tensor(pos).to(self.device)
        pos = pos.unsqueeze(0).expand_as(ent_idx)
        ents = ents + self.pos_embeddings(pos)

        concat_input = torch.cat((r, ents), dim=1)

        pos_query = self.edge_query_embedding(pos)
        pos_key = self.edge_key_embedding(pos)
        pos_value = self.edge_value_embedding(pos)

        x = self.preprocess(concat_input)
        x = self.decoder[int(ent_idx.shape[1]) - 1](x, miss_ent_domain, pos_query, pos_key, pos_value)[:, -1, :]

        miss_ent_domain = torch.LongTensor([miss_ent_domain - 1]).to(self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = self.ent_embeddings + mis_pos

        scores = torch.mm(x, tar_emb.transpose(0, 1))
        scores += self.b.expand_as(scores)
        return scores, 0
