import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, output_dim=None, num_layers=2, nhead=2, dropout=0.1, pooling="last"):
        super(TemporalEncoder, self).__init__()
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        
        self.input_proj = nn.Linear(1, hidden_dim)  
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

        if pooling == "attention":
            self.attention_vector = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        
        if x.dim() != 4:
            raise ValueError("TemporalEncoder expects input with shape [B, num_windows, num_metrics, seq_len]")

        bs, num_windows, num_metrics, seq_len = x.shape


        x = x.view(bs * num_windows, num_metrics, seq_len, 1)

        x = self.input_proj(x)

  
        x = x.view(bs * num_windows * num_metrics, seq_len, self.hidden_dim)

      
        outputs = []
        chunk_size = 4096 
        for i in range(0, x.size(0), chunk_size):
            sub_x = x[i:i + chunk_size]
            sub_x = self.transformer_encoder(sub_x)
            outputs.append(sub_x)
        x = torch.cat(outputs, dim=0)

     
        if self.pooling == "last":
            x = x[:, -1, :] 
        elif self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "attention":
            attn_scores = torch.matmul(x, self.attention_vector)
            attn_weights = F.softmax(attn_scores, dim=1)
            x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        x = x.view(bs, num_windows, num_metrics, -1)
        x = self.output_proj(x) 
        return x