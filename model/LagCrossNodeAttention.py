import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LagCrossNodeAttention(nn.Module):
    
    def __init__(self, d_in: int, d_out: int = 64, num_heads: int = 4, num_lags: int = 3, dropout: float = 0.0, device=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.h = num_heads
        self.K = num_lags
        self.dh = d_out // num_heads
        self.device = device

        self.Wq = nn.Linear(d_in, self.dh * self.h, bias=False)
        self.Wk = nn.Linear(d_in, self.dh * self.h, bias=False)
        self.Wv = nn.Linear(d_in, self.dh * self.h, bias=False)
        self.Wo = nn.Linear(self.dh * self.h, d_out, bias=False)

        self.rho = nn.Parameter(torch.randn(self.K, self.dh) * 0.01)
        self.beta = nn.Parameter(torch.linspace(-0.25, -0.25 * self.K, steps=self.K))
        
        self.gate_w = nn.Linear(d_in + d_out, 1)
        self.ffn = nn.Sequential(
            nn.Linear(d_in + d_out, 2 * d_out),
            nn.ReLU(inplace=True),
            nn.Linear(2 * d_out, d_out),
        )
        self.ln = nn.LayerNorm(d_out)
        
        self.gamma_A = nn.Parameter(torch.tensor(3.0))
        
        self.lambda_backdoor = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout)
        
        self.last_align_loss = torch.tensor(0.0)

    def forward(self, H: torch.Tensor, A_stack: torch.Tensor, mask: torch.Tensor = None):

        B, T, N, Din = H.shape
        device = H.device
        K = min(self.K, T - 1)
        
        if K <= 0:
            proj = self.Wo(self.Wv(H).view(B, T, N, self.h, self.dh).flatten(-2))
            self.last_align_loss = torch.tensor(0.0, device=device)
            return proj

        Q = self.Wq(H).view(B, T, N, self.h, self.dh)
        Kx = self.Wk(H).view(B, T, N, self.h, self.dh)
        Vx = self.Wv(H).view(B, T, N, self.h, self.dh)

        H_mean = H.mean(dim=2, keepdim=True)
        C_Q = self.Wq(H_mean).view(B, T, 1, self.h, self.dh)
        C_K = self.Wk(H_mean).view(B, T, 1, self.h, self.dh)

        eps = 1e-8
        if A_stack.dim() == 3:
            A = A_stack[:K].unsqueeze(0).expand(B, -1, -1, -1) 
        else:
            A = A_stack[:, :K]

        per_tau_msgs = []        
        per_tau_logits_mean = [] 
        spatial_attn_dists = [] 

        Tr = T - K 

        for tau in range(1, K + 1):
            Q_tau = Q[:, K:, :, :, :]                  
            K_tau = Kx[:, K-tau : T-tau, :, :, :] + self.rho[tau-1] 
            V_tau = Vx[:, K-tau : T-tau, :, :, :]

            logits = torch.einsum('btihd, btjhd -> btihj', Q_tau, K_tau) / math.sqrt(self.dh)
            logits = logits + self.beta[tau-1] 

            C_Q_tau = C_Q[:, K:, :, :, :]
            C_K_tau = C_K[:, K-tau : T-tau, :, :, :] + self.rho[tau-1]
           
            proj_q = (Q_tau * C_Q_tau).sum(dim=-1)
            proj_k = (K_tau * C_K_tau).sum(dim=-1)
    
            confounder_score = torch.einsum('btih, btjh -> btihj', proj_q, proj_k) / math.sqrt(self.dh)
  
            logits = logits - self.lambda_backdoor * confounder_score


            A_tau = A[:, tau-1] 
            
        
            bias_causal = torch.log(A_tau + eps).view(B, 1, N, 1, N)
            logits = logits + self.gamma_A * bias_causal
            
            if mask is not None:
                mask_bool = (mask > 0).view(1, 1, N, 1, N)
                logits = logits.masked_fill(~mask_bool, -1e9)

            omega = torch.softmax(logits, dim=-1) 
            
            spatial_attn_dists.append(omega.mean(dim=3))

            lag_score = logits.mean(dim=(-1, -2)) 
            per_tau_logits_mean.append(lag_score)

            msg = torch.einsum('btihj, btjhd -> btihd', omega, V_tau)
            per_tau_msgs.append(msg)

        msgs = torch.stack(per_tau_msgs, dim=3)               
        lag_scores = torch.stack(per_tau_logits_mean, dim=-1) 
        
        pi = torch.softmax(lag_scores, dim=-1)            

        pi_exp = pi.view(B, Tr, N, K, 1, 1)
        msg_all = (msgs * pi_exp).sum(dim=3)              
        
        msg_all = msg_all.contiguous().view(B, Tr, N, self.h * self.dh)
        msg_all = self.Wo(msg_all)
        msg_all = self.dropout(msg_all)

        base = H[:, K:, :, :]
        # Gating
        z = torch.sigmoid(self.gate_w(torch.cat([base, msg_all], dim=-1)))
        fused = self.ln(base + self.ffn(torch.cat([base, z * msg_all], dim=-1)))

        H_out = H.new_zeros(B, T, N, self.d_out)
        if self.d_in == self.d_out:
             H_out[:, :K] = H[:, :K]
        else:
             H_out[:, :K] = self.Wo(self.Wv(H[:, :K]).view(B, K, N, self.h, self.dh).flatten(-2))
        H_out[:, K:] = fused

        A_temporal_mass = A.sum(dim=-1) 
       
        A_temporal_mass = A_temporal_mass.transpose(1, 2)
        
        A_temporal_dist = A_temporal_mass / (A_temporal_mass.sum(dim=-1, keepdim=True) + eps)
        A_pi_target = A_temporal_dist.unsqueeze(1).expand_as(pi) 
        
        loss_temporal = F.kl_div((pi + eps).log(), A_pi_target, reduction='batchmean')

        omega_stack = torch.stack(spatial_attn_dists, dim=3) 
        
        A_spatial_target = A / (A.sum(dim=-1, keepdim=True) + eps)
        
        A_spatial_target = A_spatial_target.permute(0, 2, 1, 3)
        
        A_spatial_target = A_spatial_target.unsqueeze(1).expand_as(omega_stack)
        
        loss_spatial = F.kl_div((omega_stack + eps).log(), A_spatial_target, reduction='batchmean')

        self.last_align_loss = 0.5 * loss_temporal + 0.5 * loss_spatial

        return H_out