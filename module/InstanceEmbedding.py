
import torch
import torch.nn as nn
import torch.nn.functional as F

class InstanceEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,              
        hidden_dim,             
        agg_type="timewise_attn",
        d_model=64,           
        dropout=0.0
    ):
        super().__init__()
        self.agg_type = agg_type
        self.d_model = d_model

        if agg_type == "timewise_attn":
            
            self.metric_proj = nn.Linear(input_dim, d_model)
         
            self.score = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            )
        elif agg_type == "timewise_mean":
            self.metric_proj = nn.Linear(input_dim, d_model)
        else:
            raise ValueError(f"Unsupported agg_type: {agg_type}")

    def forward(self, x, mapping=None):
        if mapping is None:
            raise ValueError("mapping is required for InstanceEndedding forward")

        if x.dim() != 4:
            raise ValueError("x must be [B, W, M_total, D_in]")

        B, W, M_total, _ = x.shape
        x_flat = x.reshape(B * W, M_total, -1)
        z = self._aggregate_flat(x_flat, mapping)  # [B*W, N, d_model]
        return z.reshape(B, W, z.size(1), z.size(2))

    def _aggregate_flat(self, x, mapping):
        B, M_total, _ = x.shape
        pods = []
        start = 0
        counts = list(mapping.values())
        if sum(counts) != M_total:
            raise ValueError("Sum of mapping metric counts must equal M_total")

        for m_i in counts:
            seg = x[:, start:start + m_i, :] 
            start += m_i

            seg_proj = self.metric_proj(seg)

            if self.agg_type == "timewise_attn":
                logits = self.score(seg_proj).squeeze(-1)
                attn = torch.softmax(logits, dim=1)
                agg = torch.sum(seg_proj * attn.unsqueeze(-1), dim=1)  
            elif self.agg_type == "timewise_mean":
                agg = seg_proj.mean(dim=1)
            else:
                raise ValueError(f"Unsupported agg_type: {self.agg_type}")

            pods.append(agg.unsqueeze(1)) 

        return torch.cat(pods, dim=1)