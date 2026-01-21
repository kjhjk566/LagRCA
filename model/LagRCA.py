import torch
import torch.nn as nn
import torch.nn.functional as F
from module.InstanceEmbedding import InstanceEmbedding

from module.NodeDecoder import NodeDecoder
from module.TemporalEncoder import TemporalEncoder
from model.MultiLagCausalLearner import MultiLagCausalLearner
from model.LagCrossNodeAttention import LagCrossNodeAttention
import math

class LagRCA(nn.Module):
    def __init__(self,config, input_dim, hidden_dim, sca_hidden_dim,metric_num, temperature=0.5, lambda_reg=0.001,lambda_granger = 0.5,lambda_sparse = 1.0,device = None,adj = None):
       
        super(LagRCA, self).__init__()
        self.transformer_encoder = TemporalEncoder(hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2, nhead=2, dropout=0.1)
        self.pod_embedding = InstanceEmbedding(input_dim, hidden_dim)
        
        self.config = config
        self.lambda_granger = lambda_granger
        self.lambda_sparse = lambda_sparse

        self.decoder = NodeDecoder(node_input_dim=64, hidden_dim=128, node_mapping=config.instance_metric_count_dict)  # 实例解码器

        self.output_layer = nn.Linear(sca_hidden_dim, 1)

        num_nodes = len(config.all_enum.keys())
        self.lag_K = 3
        self.num_heads = 4
        self.pod_hidden_dim = hidden_dim
        self.d_model_out = 64
        if adj is not None:
            prior_mask = torch.tensor(adj, dtype=torch.float32, device=device)
            prior_mask = (prior_mask > 0).float()
        else:
            prior_mask = torch.ones(num_nodes, num_nodes, dtype=torch.float32, device=device)
        # forbid self loops on same-time
        eye = torch.eye(num_nodes, device=device)
        prior_mask = prior_mask * (1.0 - eye)
        self.register_buffer("prior_mask", prior_mask)

        
        self.causal_learner = MultiLagCausalLearner(
            num_nodes=num_nodes,
            input_dim=self.pod_hidden_dim,
            num_lags=self.lag_K,
            rank=8,
            mask=self.prior_mask,
            device=device
        )
     

        self.lcna = LagCrossNodeAttention(
            d_in=self.pod_hidden_dim,
            d_out=self.d_model_out,
            num_heads=self.num_heads,
            num_lags=self.lag_K,
            dropout=0.1,
            device=device
        )
        self.lambda_align = 0.001

    def forward(self, x_window, y_window):
        
        p = self.get_prediction(x_window)  # 获取预测结果
        loss = self.loss(p, y_window)  # 计算损失
        return loss

    def get_prediction(self, x_window):
       
     
        metric_embedding = self.transformer_encoder(x_window)

        pod_embedding= self.pod_embedding(x_window,self.config.instance_metric_count_dict) 
      
        
        A_list = self.causal_learner(pod_embedding)          
        
        st_time = self.lcna(pod_embedding, A_list, self.prior_mask)  
        
        st_out = st_time[:, -1, :, :]                           
        
        instance_names = list(self.config.instance_metric_count_dict.keys())
        metric_embedding = metric_embedding[:, -1, :, :]
        predictions = self.decoder.forward(h_instances=st_out, metric_embeddings=metric_embedding, instance_names=instance_names)  # [B, total_metrics]
        return predictions

    def loss(self, predictions, y_window):
        device = predictions.device
        task_loss = F.mse_loss(predictions, y_window)
        align_loss = getattr(self.lcna, "last_align_loss", torch.tensor(0.0, device=device))

        reg_l1 = torch.tensor(0.0, device=device)
        reg_prior = torch.tensor(0.0, device=device)
        reg_dag = torch.tensor(0.0, device=device)
        reg_ent = torch.tensor(0.0, device=device)
        reg_resid = torch.tensor(0.0, device=device)

        if hasattr(self, "causal_learner") and self.causal_learner is not None:
            cl = self.causal_learner
            if hasattr(cl, "get_regularization_loss"):
                reg_l1 = cl.get_regularization_loss(cl.get_last_graph())
            if hasattr(cl, "dag_penalty"):
                reg_dag = cl.dag_penalty()
            if hasattr(cl, "get_last_graph"):
                A_last = cl.get_last_graph()
                eps = 1e-12
                P = A_last.clamp_min(eps)
                row_ent = -(P * P.log()).sum(dim=-1)
                reg_ent = row_ent.mean()
            if hasattr(cl, "residual_reg"):
                reg_resid = cl.residual_reg()
            A_last = cl.get_last_graph()
            prior = self.prior_mask.unsqueeze(0)
            reg_prior = F.l1_loss(A_last, prior)
    
        lambda_l1    = 1e-4
        lambda_prior = 1e-1
        lambda_dag   = 1e-3
        lambda_ent   = 1e-4
        lambda_align = 1e-3
        lambda_resid = 1e-4

        loss = (
            task_loss
            + lambda_align * align_loss
            + lambda_l1 * reg_l1
            + lambda_prior * reg_prior
            + lambda_dag * (reg_dag ** 2)
            + lambda_ent * reg_ent
            + lambda_resid * reg_resid
        )

        return loss

    def get_ans(self, x_window, y_window):
        metric_losses, instance_losses, total_loss = self.calculate_instance_loss(
            predictions=x_window, 
            y_true=y_window, 
            instance_metric_count_dict=self.config.instance_metric_count_dict,
            instance_names=list(self.config.all_enum.keys())
        )
        top5_indices = torch.topk(instance_losses, k=5, dim=1).indices
        top5_instances = [list(self.config.all_enum.keys())[i] for i in top5_indices[0]]

        return top5_instances



    def calculate_metric_loss(self, x_window, y_window, loss_type='mse'):
        
        predictions = self.get_prediction(x_window)
        y_true = y_window
        y_true = y_true.unsqueeze(0) 
      
        if loss_type == 'mse':
            metric_losses = F.mse_loss(predictions, y_true, reduction='none')
        elif loss_type == 'mae':
            metric_losses = F.l1_loss(predictions, y_true, reduction='none')
        elif loss_type == 'rmse':
            metric_losses = torch.sqrt(F.mse_loss(predictions, y_true, reduction='none'))
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return metric_losses