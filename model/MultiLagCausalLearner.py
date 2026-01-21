import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLagCausalLearner(nn.Module):
    
    def __init__(self, num_nodes, input_dim, num_lags=5, rank=8, mask=None, device='cpu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_lags = num_lags
        self.device = device
        self.rank = rank

       
        if mask is None:
            mask = torch.ones(num_nodes, num_nodes, device=device)
        self.register_buffer("mask", mask.float())

        # Context Extractor (Shared by both Structure and Strength)
        self.hnet_hidden = 64
        self.context_mapper = nn.Sequential(
            nn.Linear(input_dim, self.hnet_hidden),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.dyn_struct_U = nn.Linear(self.hnet_hidden, num_lags * num_nodes * rank)
        self.dyn_struct_V = nn.Linear(self.hnet_hidden, num_lags * num_nodes * rank)
        
        self.static_struct_logits = nn.Parameter(torch.randn(num_lags, num_nodes, num_nodes) * 0.1 - 5.0)
        
        self.dyn_weight_U = nn.Linear(self.hnet_hidden, num_lags * num_nodes * rank)
        self.dyn_weight_V = nn.Linear(self.hnet_hidden, num_lags * num_nodes * rank)
        
        self.static_weight_U = nn.Parameter(torch.randn(num_lags, num_nodes, rank) * 0.01)
        self.static_weight_V = nn.Parameter(torch.randn(num_lags, num_nodes, rank) * 0.01)
        
        self.last_A = None

    def gumbel_sigmoid(self, logits, tau=1.0, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = torch.sigmoid(gumbels)

        if hard:
            y_hard = (y_soft > 0.5).float()
            ret = y_hard - y_soft.detach() + y_soft
            return ret
        else:
            return y_soft

    def forward(self, H, temperature=1.0, hard=False):
        
        B, T, N, D = H.shape
        
       
        c = H.mean(dim=(1, 2)) 
        ctx_emb = self.context_mapper(c) 

       
        ds_U = self.dyn_struct_U(ctx_emb).view(B, self.num_lags, N, self.rank)
        ds_V = self.dyn_struct_V(ctx_emb).view(B, self.num_lags, N, self.rank)
        
      
        dyn_logits = torch.matmul(ds_U, ds_V.transpose(-1, -2)) # [B, K, N, N]
        
       
        logits = self.static_struct_logits.unsqueeze(0) + dyn_logits
        
        mask_expanded = self.mask.view(1, 1, N, N)
        
        logits = logits.masked_fill(mask_expanded == 0, -1e9)
       
        eye = torch.eye(N, device=self.device).view(1, 1, N, N)
        logits = logits.masked_fill(eye == 1, -1e9)
        
      
        M = self.gumbel_sigmoid(logits, tau=temperature, hard=hard) # [B, K, N, N] \in {0, 1} (approx)

       
        dw_U = self.dyn_weight_U(ctx_emb).view(B, self.num_lags, N, self.rank)
        dw_V = self.dyn_weight_V(ctx_emb).view(B, self.num_lags, N, self.rank)
        
        
        final_U = self.static_weight_U.unsqueeze(0) + dw_U
        final_V = self.static_weight_V.unsqueeze(0) + dw_V
        
        
        W = F.softplus(torch.matmul(final_U, final_V.transpose(-1, -2))) # [B, K, N, N]

        
        A_stack = M * W 
        
        self.last_A = A_stack
        
        return A_stack

    def get_last_graph(self):
        return self.last_A

    def get_regularization_loss(self, A_stack):
        
        loss_l1 = torch.mean(torch.abs(A_stack))
        
        return loss_l1