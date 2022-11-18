import torch.nn as nn
import torch


class MeanPooling(nn.Module):
    def __init__(self, config=None):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self, config=None):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self, config=None):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


# testing not good
# L 维度
class MeanMaxPooling(nn.Module):
    def __init__(self, config=None):
        super(MeanMaxPooling, self).__init__()
        self.mean_pooling = MeanPooling()
        self.max_pooling = MaxPooling()

    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        max_pooling_embeddings = self.max_pooling(last_hidden_state, attention_mask)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        # logits = nn.Linear(hidden_size*2, 1)(mean_max_embeddings) # twice the hidden size
        return mean_max_embeddings

        #print(f'Last Hidden State Output Shape: {last_hidden_state.detach().numpy().shape}')
        #print(f'Mean-Max Embeddings Output Shape: {mean_max_embeddings.detach().numpy().shape}')
        #print(f'Logits Shape: {logits.detach().numpy().shape}')


### https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
# layer 维度
class WeightedLayerPooling(torch.nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None, logger=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else torch.nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        ) #  size: #(num_layers), learnable
        if logger is not None:
            logger.info("Attention weighted layer num: ", layer_weights.shape[0])

    def forward(self, all_hidden_states):
        # layer 越大表示越接近输出
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()  # 以layer维度，将各layer相加
        return weighted_average

# hidden_state作attention, 来确定各个L的权重
# 消去L
class AttentionPooling(nn.Module):
    def __init__(self, config):
        in_dim = config.hidden_size
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x, mask):
        # x size: #(b, L, H)
        w = self.attention(x).float() #
        w[mask==0]=float('-inf')
        w = torch.softmax(w,1)
        x = torch.sum(w * x, dim=1) # L维度求和
        return x


class AttentionMeanPooling(nn.Module):
    def __init__(self, config):
        super(AttentionMeanPooling, self).__init__()
        self.mean_pooler = MeanPooling(config)
        self.attention_pooler = AttentionPooling(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, last_hidden_state, attention_mask):
        # (B, hidden)
        mean_pooling_embeddings = self.mean_pooler(last_hidden_state, attention_mask)
        attention_pooling_embeddings = self.attention_pooler(last_hidden_state, attention_mask)
        attention_mean_embeddings = torch.cat((self.ln_1(mean_pooling_embeddings), self.ln_2(attention_pooling_embeddings)), 1)
        # logits = nn.Linear(hidden_size*2, 1)(mean_max_embeddings) # twice the hidden size
        return attention_mean_embeddings


class AttentionWeightedPooling(nn.Module):
    def __init__(self, config, layer_start: int = 10, layer_weights=None, logger=None):
        super(AttentionWeightedPooling, self).__init__()
        self.num_hidden_layers = 12
        self.layer_start = layer_start
        self.layer_num = self.num_hidden_layers - self.layer_start + 1
        self.poolers = nn.ModuleList([AttentionPooling(config)] * self.layer_num)
        self.layer_weights = layer_weights if layer_weights is not None \
            else torch.nn.Parameter(
            torch.tensor([1] * self.layer_num, dtype=torch.float)
        )  #  size: #(num_layers), learnable
        if logger is not None:
            logger.info("Attention weighted layer num: " + str(self.layer_num))

    def forward(self, all_hidden_states, mask):
        # layer 越大表示越接近输出
        #print(all_hidden_states.size())
        # [hidden_layers + 1, b, l, h]
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        all_poolings = list(self.poolers[i](all_layer_embedding[i, :, :, :], mask) for i in range(self.layer_num))   # generator to list
        all_layer_embedding = torch.stack(all_poolings)  # concat use old dimension, stack use new dimension
        # print(all_layer_embedding.size())

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()  # 以layer维度，将各layer weighted相加
        return weighted_average

