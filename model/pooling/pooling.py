import torch.nn as nn
import torch


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else torch.nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        ) #  size: #(num_layers), learnable

    def forward(self, all_hidden_states):
        # layer 越大表示越接近输出
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()  # 以layer维度，将各layer相加
        return weighted_average

# hidden_state作attention, 来确定各个L的权重
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
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




