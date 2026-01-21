    
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeDecoder(nn.Module):
    def __init__(self, node_input_dim, hidden_dim, node_mapping,metric_embbeding_dim=64):
       
        super(NodeDecoder, self).__init__()
        self.node_mapping = node_mapping
        self.hidden_dim = hidden_dim
        self.node_input_dim = node_input_dim
        self.fc1 = nn.Linear(node_input_dim, hidden_dim)
        
       
        self.instance_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
       
        self.metric_predictor = nn.Sequential(
            nn.Linear(node_input_dim + metric_embbeding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.predictor_gnet = nn.Sequential(
            nn.Linear(node_input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def predict_next_step_gnet(self, x):
        
        out = self.predictor_gnet(x)
        out = out.squeeze(-1)
        return out
    def forward(self, h_instances, instance_names, metric_embeddings):
        B, N, _ = h_instances.shape
        B_metric, total_metrics, metric_dim = metric_embeddings.shape
        
        assert B == B_metric, f"Batch sizes don't match: {B} vs {B_metric}"
        
       
        encoded_instances = h_instances 
        
    
        batch_predictions = []
        
        for b in range(B):
            metric_predictions = []
            start_idx = 0
            
            for i, instance_name in enumerate(instance_names):
                num_metrics = self.node_mapping.get(instance_name, 0)
                if num_metrics == 0:
                    continue
                
               
                instance_feature = encoded_instances[b, i] 
                
               
                end_idx = start_idx + num_metrics
                if end_idx > total_metrics:
                    print(f"Warning: end_idx {end_idx} > total_metrics {total_metrics}")
                    break
                    
                instance_metric_embeddings = metric_embeddings[b, start_idx:end_idx]  # [num_metrics, 64]
                
             
                expanded_instance_feature = instance_feature.unsqueeze(0).repeat(num_metrics, 1)  # [num_metrics, hidden_dim]
                
       
                combined_features = torch.cat([expanded_instance_feature, instance_metric_embeddings], dim=1)  # [num_metrics, hidden_dim + 64]
                
             
                metric_preds = self.metric_predictor(combined_features)  # [num_metrics, 1]
                metric_predictions.append(metric_preds.squeeze(-1))  # [num_metrics]
                
                start_idx = end_idx
            
        
            if metric_predictions:
                batch_pred = torch.cat(metric_predictions, dim=0) 
            else:
                batch_pred = torch.zeros(total_metrics, device=h_instances.device)
            
        
            if batch_pred.shape[0] < total_metrics:
                padding = torch.zeros(total_metrics - batch_pred.shape[0], device=h_instances.device)
                batch_pred = torch.cat([batch_pred, padding], dim=0)
            
            batch_predictions.append(batch_pred)
        
     
        final_predictions = torch.stack(batch_predictions, dim=0)  
        
        return final_predictions
    
    def forward_padded(self, h_instances, instance_names):
        
        batch_predictions = self.forward(h_instances, instance_names)
        
     
        max_total_metrics = 0
        for b in range(len(batch_predictions)):
            total_metrics = sum(len(pred) for pred in batch_predictions[b])
            max_total_metrics = max(max_total_metrics, total_metrics)
        
        if max_total_metrics == 0:
            return torch.zeros((len(batch_predictions), 0), device=h_instances.device)
        
    
        padded_predictions = []
        for b in range(len(batch_predictions)):
          
            batch_pred = torch.cat(batch_predictions[b]) if batch_predictions[b] else torch.tensor([], device=h_instances.device)
            
          
            if len(batch_pred) < max_total_metrics:
                pad_size = max_total_metrics - len(batch_pred)
                padding = torch.zeros(pad_size, device=h_instances.device)
                batch_pred = torch.cat([batch_pred, padding])
            
            padded_predictions.append(batch_pred)
        
        return torch.stack(padded_predictions, dim=0)  