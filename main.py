from ast import arg
import sys
import os
import time


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import argparse

from config import Config

from module.DataProcessor import DataProcessor,TimeWindowDataset

from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader
from module.NodeDecoder import NodeDecoder

from model.LagRCA import LagRCA
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle
from module.RootCauseScorer import RootCauseScorer



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('-ds','--dataset', type=str, default="D1")
   
    return parser.parse_args()

def evaluate_topk_accuracy(all_ans, all_labels, topk_list=[1,3, 5]):
    results = {k: 0 for k in topk_list}
    total = len(all_ans)

    for pred_list, label in zip(all_ans, all_labels):
        # 确保 label 是列表格式
        label = [label] if isinstance(label, str) else label
        for k in topk_list:
            topk_pred = pred_list[:k]
            hit = any(
                any(p.startswith(l) for p in topk_pred) for l in label
            )
            if hit:
                results[k] += 1

    return {k: results[k] / total for k in topk_list}



if __name__ == "__main__":
 
    args = parse_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
   
    args.data_path = os.path.join(base_dir, 'data', args.dataset)
    
    
    config = Config(args.dataset)
    config.batch_size = args.batch_size
    
    data_path = os.path.join(args.data_path, 'normal_data.pkl')
    adj_path = os.path.join(args.data_path, 'adj.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(adj_path, 'rb') as f:
        adj = pickle.load(f)

    # Data Processing
    data_processor = DataProcessor(data, config, window_size=args.window_size, stride=args.stride)
    config.instance_metric_count_dict = data_processor.instance_metric_count_dict
    
    train_dataset = TimeWindowDataset(data_processor, config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    first_batch = next(iter(train_loader))
    x_window = first_batch[0]
    y_window = first_batch[2]
    metric_num = y_window.shape[1]
    input_dim = x_window.shape[-1]
 
    model = LagRCA(config, device=device, adj=adj, metric_num=metric_num, 
                input_dim=input_dim, hidden_dim=64, sca_hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    root_cause_scorer = RootCauseScorer(alpha=1.0, beta=0.01, config=config, device=device)

    
    print("\nStart Training...")
    model.train()
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        epoch_loss = 0.0
        batch_count = 0
        
        for x_window, instance_names, y_window in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x_window = x_window.to(device)
            y_window = y_window.to(device)
            
            optimizer.zero_grad()
            loss = model(x_window, y_window)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    model.eval()
    
    case_path = os.path.join(args.data_path, 'case_data.pkl')
    if not os.path.exists(case_path):
        print(f"Test case file not found: {case_path}")
        sys.exit(1)
    else:
        with open(case_path, 'rb') as f:
            test_case = pickle.load(f)

        all_ans = []
        all_labels = []
        results_data = []

        print("\nStart Testing...")
        
        for case_id, (case_data, label, ts) in enumerate(tqdm(test_case)):
            all_labels.append(label)

            if case_data.shape[0] < args.window_size:
               
                all_ans.append([]) 
                continue

            case_processor = DataProcessor(case_data, config, window_size=args.window_size, stride=args.stride)
            test_dataset = TimeWindowDataset(case_processor, config)
            test_loader_case = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            all_score = None
            for x_window, _, y_window in test_loader_case:
                x_window = x_window.to(device)
                y_window = y_window[0].to(device)
                
                with torch.no_grad():
                    anomaly_score = model.calculate_metric_loss(x_window, y_window)
                
                if all_score is None:
                    all_score = anomaly_score
                else:
                    all_score += anomaly_score
            
            if len(test_loader_case) > 0:
                all_score = all_score / len(test_loader_case)
                all_score = all_score.squeeze(-1).cpu()

                A_list = model.causal_learner.get_last_graph()
                
                A_list = A_list.squeeze(0)  
                ans, _, _ = root_cause_scorer.get_root_cause_by_upstream_adjustment(all_score, A_list, label, beta=0.5)
            else:
                ans = []
            all_ans.append(ans)
            
        # Evaluate
        accuracy = evaluate_topk_accuracy(all_ans, all_labels, topk_list=[1, 5])
        print("\nEvaluation Results:")
        print(f"Top-1 Accuracy: {accuracy[1]:.2%}")
        print(f"Top-5 Accuracy: {accuracy[5]:.2%}")

