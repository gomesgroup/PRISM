import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Mapping, Sequence, Tuple, Union

class AmidePredictor(nn.Module):

    def __init__(self,
            graph_in_dim: int,
            n_graph_layers: int,
            n_output_layers: int,
            use_control: bool = True,
            n_additional_features: int = 1,
            # acid_index: list = [],
            # amine_index: list = [],
            # int_index: list = []
            ):
        super(AmidePredictor, self).__init__()

        self.use_control = use_control
        self.n_additional_features = n_additional_features
        # self.acid_index = acid_index
        # self.amine_index = amine_index
        # self.int_index = int_index
        
        acid_layers = []
        amine_layers = []
        int_layers = []

        for _ in range(n_graph_layers):
            acid_layers.append(GraphAttentionLayer(graph_in_dim, graph_in_dim))
            amine_layers.append(GraphAttentionLayer(graph_in_dim, graph_in_dim))
            int_layers.append(GraphAttentionLayer(graph_in_dim, graph_in_dim))

        self.acid_model = nn.ModuleList(acid_layers)
        self.amine_model = nn.ModuleList(amine_layers)
        self.int_model = nn.ModuleList(int_layers)

        # Calculate input dimension for output layers
        # graph_in_dim*2 for molecular features (acid, amine) + n_additional_features
        # graph_in_dim*3 for molecular features (acid, amine, int1) + n_additional_features
        n_molecular_features = 3
        output_input_dim = graph_in_dim * n_molecular_features
        if self.use_control:
            output_input_dim += self.n_additional_features  # Add all additional features
        
        out_layers = []
        for _ in range(n_output_layers):
            out_layers.append(nn.Linear(output_input_dim, output_input_dim))
            out_layers.append(nn.GELU())
        out_layers.append((nn.Linear(output_input_dim, graph_in_dim)))
        out_layers.append(nn.GELU())
        out_layers.append((nn.Linear(graph_in_dim, 1)))
        
        self.out_layers = nn.ModuleList(out_layers)


    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # Use full molecular features (a, q, aim)
        acid_q_expanded = data['acid_q'].unsqueeze(-1)  # (n_atoms, 1)
        acid_feat = torch.cat([data['acid_a'], data['acid_aim'], acid_q_expanded], dim=-1)  # (n_atoms, 513)
        # acid_feat = data['acid_aim']  # Use only AIM features
        
        amine_q_expanded = data['amine_q'].unsqueeze(-1)  # (n_atoms, 1) 
        amine_feat = torch.cat([data['amine_a'], data['amine_aim'], amine_q_expanded], dim=-1)  # (n_atoms, 513)
        # amine_feat = data['amine_aim']  # Use only AIM features
        
        int_q_expanded = data['int_q'].unsqueeze(-1)  # (n_atoms, 1)
        int_feat = torch.cat([data['int_a'], data['int_aim'], int_q_expanded], dim=-1)  # (n_atoms, 513)
        # int_feat = data['int_aim']  # Use only AIM features

        # Process through graph attention layers
        for _, layer in enumerate(self.acid_model):
            acid_feat = layer(acid_feat)

        for _, layer in enumerate(self.amine_model):
            amine_feat = layer(amine_feat)
            
        for _, layer in enumerate(self.int_model):
            int_feat = layer(int_feat)

        # Combine molecular features
        # x = torch.cat([acid_feat.sum(0), amine_feat.sum(0)], dim=-1)  # (graph_in_dim*2,)
        x = torch.cat([acid_feat.sum(0), amine_feat.sum(0), int_feat.sum(0)], dim=-1)  # (graph_in_dim*3,)
        # x = torch.cat([acid_feat[self.acid_index], amine_feat[self.amine_index], int_feat[self.int_index]], dim=-1) # size 256*3
        
        
        # Add additional features if enabled
        if self.use_control:
            additional_feats = []
            
            # Collect all additional features (non-graph features)
            graph_keys = ['acid_a', 'acid_q', 'acid_aim', 'amine_a', 'amine_q', 'amine_aim',
                         'int_a', 'int_q', 'int_aim', 
                         'rate', 'pred_rate']
            
            for key, value in data.items():
                if key not in graph_keys:
                    # Process each additional feature
                    feat = value.to(device=x.device, dtype=torch.float32)
                    if feat.dim() == 0:  # scalar
                        feat = feat.unsqueeze(0)  # (1,)
                    elif feat.dim() == 2:  # (1, 1)
                        feat = feat.squeeze()  # (1,) or scalar
                        if feat.dim() == 0:
                            feat = feat.unsqueeze(0)  # (1,)
                    additional_feats.append(feat)
            
            # Concatenate all additional features
            if additional_feats:
                additional_tensor = torch.cat(additional_feats, dim=-1)
                # Ensure both tensors are float32 and on same device before concatenation
                x = x.to(torch.float32)
                x = torch.cat([x, additional_tensor], dim=-1)
        
        # Process through output layers
        for _, layer in enumerate(self.out_layers):
            x = layer(x)
        
        data['pred_rate'] = x
        return data

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:

        N = x.size(0)
        h = self.W(x)  # (N, out_dim)
        # Compute attention scores
        h_exp = h.unsqueeze(1).expand(N, N, -1)  # (N, N, out_dim)
        h_concat = torch.cat([h_exp, h_exp.transpose(0, 1)], dim=-1)  # (N, N, 2*out_dim)
        e = F.leaky_relu(self.a(h_concat)).squeeze(-1)  # (N, N)

        # Normalize attention scores
        alpha = F.softmax(e, dim=1)  # (N, N)

        # Weighted sum of node features
        out = torch.matmul(alpha, h)  # (N, out_dim)
        return out

