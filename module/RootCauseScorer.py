import torch
import torch.nn as nn
import torch.nn.functional as F

class RootCauseScorer(nn.Module):

    """
    Root Cause Localization Module: Calculates root cause score based on anomaly degree and propagation impact.
    """

    def __init__(self, alpha=1.0, beta=0.4, config=None, loss_type='mse', device='cpu'):
        super(RootCauseScorer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk = config.topk
        self.config = config
        self.device = device
        self.metric_topk = 2
        self.instance_metric_count_dict = config.instance_metric_count_dict
        
        # Use instance order consistent with configuration
        if hasattr(config, 'instance_metric_names'):
            self.instance_names = list(config.instance_metric_names.keys())
            self.instance_metric_names_dict = config.instance_metric_names
        else:
            self.instance_names = list(config.instance_metric_count_dict.keys())
            self.instance_metric_names_dict = None
            
        self.loss_type = loss_type

    def get_top_loss_instance_info(self, instance_losses,top_error_metrics, label_instances=None, topk=None):
        # Get top-k instance names
        if topk is None:
            topk = self.topk
        topk = min(topk, len(self.instance_names))
        _, indices = torch.topk(instance_losses, topk)
        topk_names = [self.instance_names[i] for i in indices.tolist()]
        
        # Return details of top-k instances
        topk_details = {}
        for i, instance_name in enumerate(topk_names):
            instance_idx = indices[i].item()
            topk_details[instance_name] = {
                'rank': i + 1,
                'instance_loss': instance_losses[instance_idx].item(),
                'top_error_metrics': top_error_metrics[instance_name]
            }
        
        # Analyze label instances using computed data
        label_anomaly_info = {}
        if label_instances is not None:
            if isinstance(label_instances, str):
                label_instances = [label_instances]
            
            # Rank all instances
            sorted_losses, sorted_indices = torch.sort(instance_losses, descending=True)
            
            for label_instance in label_instances:
                matched_instances = []
                for i, instance_name in enumerate(self.instance_names):
                    if instance_name.startswith(label_instance):
                        matched_instances.append((instance_name, i))
                
                if not matched_instances:
                    print(f"Warning: No match found for '{label_instance}'")
                    continue
                
                for instance_name, instance_idx in matched_instances:
                    instance_loss = instance_losses[instance_idx].item()
                    
                    rank_in_all = (sorted_indices == instance_idx).nonzero(as_tuple=True)[0].item() + 1
                    
                    label_anomaly_info[instance_name] = {
                        'instance_loss': instance_loss,
                        'rank_in_all_instances': rank_in_all,
                        'total_instances': len(self.instance_names),
                        'top_error_metrics': top_error_metrics[instance_name]
                    }
            
        return topk_details, label_anomaly_info
        


    def calculate_instance_loss(self, metric_losses):
        """
        Calculate instance-level error based on metric-level errors
        Args:
            metric_losses: [total_metrics] Error for each metric
        Returns:
            instance_losses: [num_instances] Average error for each instance
            total_loss: scalar Total loss
            top_error_metrics: dict Top 3 metrics with highest error for each instance
        """
        device = metric_losses.device
        instance_losses = []
        top_error_metrics = {}
        
        metric_losses = metric_losses.flatten()
        
        # Use configuration order
        for instance_name in self.instance_names:
            if self.instance_metric_names_dict:
                instance_info = self.instance_metric_names_dict[instance_name]
                start_idx = instance_info['start_idx']
                end_idx = instance_info['end_idx']
                metric_names = instance_info['full_names']
                num_metrics = end_idx - start_idx
            else:
                raise ValueError("Missing instance name mapping info")
            
            if num_metrics == 0:
                instance_losses.append(torch.tensor(0.0, device=device))
                top_error_metrics[instance_name] = []
            else:
                instance_metric_errors = metric_losses[start_idx:end_idx]  # [num_metrics]
                
                # Get top 3 metrics errors
                if len(instance_metric_errors) >= self.metric_topk:
                    topk_errors, topk_indices = torch.topk(instance_metric_errors, k=self.metric_topk)
                    instance_avg_loss = torch.mean(topk_errors)

                    top_metrics_info = []
                    for i, (error_val, metric_idx) in enumerate(zip(topk_errors, topk_indices)):
                        metric_name = metric_names[metric_idx.item()]
                        top_metrics_info.append({
                            'rank': i + 1,
                            'metric_name': metric_name,
                            'error_value': error_val.item(),
                            'global_idx': start_idx + metric_idx.item()
                        })
                else:
                    # Use all metrics if less than 3
                    instance_avg_loss = torch.mean(instance_metric_errors)
                    
                    top_metrics_info = []
                    sorted_errors, sorted_indices = torch.sort(instance_metric_errors, descending=True)
                    for i, (error_val, metric_idx) in enumerate(zip(sorted_errors, sorted_indices)):
                        metric_name = metric_names[metric_idx.item()]
                        top_metrics_info.append({
                            'rank': i + 1,
                            'metric_name': metric_name,
                            'error_value': error_val.item(),
                            'global_idx': start_idx + metric_idx.item()
                        })
                
                instance_losses.append(instance_avg_loss)
                top_error_metrics[instance_name] = top_metrics_info
        
        instance_losses = torch.stack(instance_losses)  # [num_instances]
        total_loss = torch.mean(metric_losses)
        return instance_losses, total_loss, top_error_metrics

    def get_ans_from_loss(self, metric_losses, A_list, label_instances=None):
        """
        Root cause localization based on prediction error
        """
        instance_losses, _, top_error_metrics = self.calculate_instance_loss(metric_losses)
        topk_details, label_anomaly_info = self.get_top_loss_instance_info(instance_losses, top_error_metrics, label_instances)
        
        _, indices = torch.topk(instance_losses, self.topk)
        topk_names = [self.instance_names[i] for i in indices.tolist()]
      
        return topk_names, topk_details, label_anomaly_info

    def get_topk(self, root_score, instance_names, topk=5):
        """
        Return top-K instance names based on root cause score
        """
        _, indices = torch.topk(root_score, topk)
        topk_names = [instance_names[i] for i in indices.tolist()]
        return topk_names

    def get_root_cause_by_walk(self, metric_losses, A_list, label_instances=None):
        instance_losses, _, top_error_metrics = self.calculate_instance_loss(metric_losses)
        topk_details, label_anomaly_info = self.get_top_loss_instance_info(instance_losses, top_error_metrics, label_instances)
        
        # Normalize instance losses
        instance_losses = instance_losses / (instance_losses.sum() + 1e-12)

        A_inst = torch.stack(A_list, 0)                         # [L, N_inst, N_inst]

        # Synthesize P
        gamma = 0.7
        L = A_inst.size(0)
        coeff = torch.tensor([gamma**(t+1) for t in range(L)], device=A_inst.device).view(L,1,1)
        P = (coeff * A_inst).sum(dim=0)                         # [N_inst, N_inst]
        P = P / (P.sum(dim=1, keepdim=True) + 1e-12)

        # Personalized PageRank
        alpha = 0.8
        r = instance_losses / (instance_losses.sum() + 1e-12)       # [N_inst]
        pi = r.clone()
       
        r = r.to(self.device)
        P = P.to(self.device)
        pi = pi.to(self.device)

        for _ in range(5):
            pi = (1-alpha) * (pi @ P) + alpha * r

        topk_idx = torch.topk(pi, k=self.config.topk).indices
        topk_instances = [self.instance_names[i] for i in topk_idx.tolist()]

        return topk_instances, topk_details, label_anomaly_info

    def get_root_cause_by_upstream_adjustment(
    self, metric_losses, A_list, label_instances=None, topk=10, gamma=None, beta=None,
    normalize='col'
    ):
        
        if beta is None:
            beta = self.beta

        instance_losses, _, top_error_metrics = self.calculate_instance_loss(metric_losses)
        r = instance_losses.to(metric_losses.device)  # [N]
        raw_r = r.clone()

        N = len(self.instance_names)
        if N == 0:
            return [], {}, {}

        device = r.device
        if isinstance(A_list, torch.Tensor):
            assert A_list.dim() == 3 and A_list.size(1) == N and A_list.size(2) == N, \
                "A_list tensor must be [K,N,N]"
            A_stack = A_list.to(device)
        else:
            A_stack = torch.stack([a.to(device) for a in A_list], dim=0)  # [K,N,N]

        if gamma is None:
            gamma = getattr(self.config, 'walk_gamma', 0.7)
        K = A_stack.size(0)
        coeff = torch.tensor([gamma ** (t + 1) for t in range(K)], device=device).view(K, 1, 1)
        W = (coeff * A_stack).sum(dim=0)  # [N, N], j->i impact (Row i, Col j)


        eps = 1e-12
        if normalize == 'col':
            # Column Normalization: Normalize out-edges for each source node j, making sum_i W_{i,j} = 1
            Z = W.sum(dim=0, keepdim=True) + eps
            W_tilde = W / Z
        elif normalize == 'row':
            # Row Normalization: Normalize in-edges for each target node i, making sum_j W_{i,j} = 1
            Z = W.sum(dim=1, keepdim=True) + eps
            W_tilde = W / Z
        elif normalize == 'softmax':
            W_tilde = torch.softmax(W, dim=0)
        else:
            W_tilde = W

     
        m = W_tilde @ r  # [N]

        s = torch.relu(r - beta * m)  # [N]


        topk = min(topk, N)
        score_vals, score_idx = torch.topk(s, k=topk)
        topk_names = [self.instance_names[i] for i in score_idx.tolist()]

        propagation_chains = self.get_propagation_chains(r, W_tilde, topk_names)

        topk_details = {}
        for rank, i in enumerate(score_idx.tolist(), start=1):
            # Upstream contribution decomposition (Check Row i)
            row = W_tilde[i, :]           # [N] In-edge weights
            contrib = row * r             # [N] Contribution of each upstream j to i
            vals, order = torch.sort(contrib, descending=True)
            upstream_contribs = []
            for rk, u in enumerate(order.tolist()[: min(10, N)], start=1):
                if vals[rk-1].item() <= 0:
                    break
                upstream_contribs.append({
                    'rank': rk,
                    'instance_name': self.instance_names[u],
                    'upstream_loss': r[u].item(),
                    'edge_weight': row[u].item(),
                    'propagated_anomaly': vals[rk-1].item(),
                })

            topk_details[self.instance_names[i]] = {
                'rank': rank,
                'instance_loss': raw_r[i].item(),
                'propagated_anomaly': m[i].item(),
                'adjusted_loss': s[i].item(),
                'upstream_contributions': upstream_contribs,
                'top_error_metrics': top_error_metrics[self.instance_names[i]],
                'propagation_chain': propagation_chains.get(self.instance_names[i], [])
            }

        _, label_anomaly_info = self.get_top_loss_instance_info(raw_r, top_error_metrics,
                                                                label_instances=label_instances, topk=topk)
        return topk_names, topk_details, label_anomaly_info
    
    def get_propagation_chains(self, instance_losses, adj_matrix, root_causes, max_depth=4):
        chains = {}
        N = len(self.instance_names)
        r = instance_losses
        W = adj_matrix # W[i, j] j -> i impact
        
        for root in root_causes:
            try:
                root_idx = self.instance_names.index(root)
            except ValueError:
                continue
                
            queue = [(root_idx, [{"name": root, "score": r[root_idx].item(), "edge": 0.0}], 1.0)]
            
            best_paths = []
            
            while queue:
                curr_idx, path, confidence = queue.pop(0)
                
                if len(path) >= max_depth:
                    best_paths.append(path)
                    continue
                
                # Check downstream (W columns)
                downstream_weights = W[:, curr_idx]
                
                # Filter by impact score
                impact_scores = downstream_weights * r
                
                top_downstream = torch.topk(impact_scores, k=min(3, N))
                
                found_child = False
                for score, child_idx in zip(top_downstream.values, top_downstream.indices):
                    child_idx = child_idx.item()
                    score = score.item()
                    edge_w = downstream_weights[child_idx].item()
                    
                 
                    if edge_w > 0.05 and r[child_idx].item() > 0.01:
                       
                        if any(node['name'] == self.instance_names[child_idx] for node in path):
                            continue
                            
                        new_node = {
                            "name": self.instance_names[child_idx],
                            "score": r[child_idx].item(),
                            "edge": edge_w
                        }
                        queue.append((child_idx, path + [new_node], confidence * edge_w))
                        found_child = True
                
                if not found_child:
                    best_paths.append(path)
            
          
            if best_paths:
              
                best_paths.sort(key=lambda p: len(p) * sum(n['score'] for n in p), reverse=True)
                top_path = best_paths[0]
                
                chain_str = ""
                for i, node in enumerate(top_path):
                    if i == 0:
                        chain_str += f"[{node['name']}(Loss:{node['score']:.2f})]"
                    else:
                        chain_str += f" --({node['edge']:.2f})--> [{node['name']}(Loss:{node['score']:.2f})]"
                chains[root] = chain_str
            else:
                chains[root] = f"[{root}] (No significant propagation found)"
                
        return chains







