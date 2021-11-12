import torch
import torch.nn as nn


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps, vocab_size, n_futures):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, layer_norm_eps)
        self.decoder_1 = nn.Linear((hidden_size + n_futures), hidden_size, bias=True)
        self.decoder_2 = nn.Linear((hidden_size), hidden_size, bias=True)
        self.decoder = nn.Linear((hidden_size), vocab_size, bias=False)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.3)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states, added_futures):
        hidden_states = self.transform(hidden_states)
        hidden_states = torch.cat((hidden_states, added_futures), 2)
        hidden_states = self.dropout_1(self.relu_1(self.decoder_1(hidden_states)))
        hidden_states = self.dropout_2(self.relu_2(self.decoder_2(hidden_states)))
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class CustumBert(nn.Module):
    def __init__(
        self,
        d_model: int,
        learning_rate: float,
        padding_idx: int,
        worst_rank: int,
        layer_eps: float,
        num_heads: int,
        n_times: int,
        n_added_futures: int,
        dropout: float,
    ):
        super().__init__()

        self.padding_idx = padding_idx
        self.worst_rank = worst_rank
        self.d_model = d_model
        self.layer_eps = layer_eps
        self.lr = learning_rate
        self.dropout = dropout
        self.n_times = n_times - 1

        self.emb = nn.Embedding(self.padding_idx + 1, self.d_model, self.padding_idx)
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.d_model,
                    num_heads=num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.n_times)
            ]
        )
        self.lns = nn.ModuleList(
            [nn.LayerNorm(self.d_model, self.layer_eps) for _ in range(self.n_times)]
        )

        self.attn_last = nn.MultiheadAttention(
            self.d_model, num_heads=num_heads, dropout=self.dropout, batch_first=True
        )
        self.time_classifier = BertLMPredictionHead(
            self.d_model, self.layer_eps, 1, n_added_futures
        )
        self.rank_classifier = BertLMPredictionHead(
            self.d_model, self.layer_eps, 1, n_added_futures
        )

        for param in self.parameters():
            param.requires_grad = True

    def update_furture_horse_vec(self, update_emb_id_before, update_emb_id_after):
        after = update_emb_id_after.view(-1).to(torch.int64)
        before = update_emb_id_before.view(-1).to(torch.int64)
        self.emb.weight.data[after] = self.emb.weight.data[before].clone()

    def forward(self, horses, covs, pad_mask):
        atten_inputs = self.emb(horses)
        for i in range(self.n_times):
            hidden_states = self.attns[i](
                atten_inputs, atten_inputs, atten_inputs, key_padding_mask=pad_mask
            )[0]
            hidden_states = self.lns[i](hidden_states)
        hidden_states = self.attn_last(
            atten_inputs, atten_inputs, atten_inputs, key_padding_mask=pad_mask
        )[0]
        time_out = self.time_classifier(hidden_states, covs)
        rank_out = self.rank_classifier(hidden_states, covs)
        return time_out, rank_out
