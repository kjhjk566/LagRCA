
import torch
import os

class Config:
    def __init__(self,data_set):
        self.data_set = data_set
        self.max_length = 512
        self.batch_size = 16
        self.topk = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        if data_set == "D1":
            self.all_enum = {'emailservice-0': 0, 'emailservice2-0': 1, 'emailservice-2': 2, 'emailservice-1': 3, 'adservice-0': 4, 'adservice2-0': 5, 'adservice-2': 6, 'adservice-1': 7, 'checkoutservice-0': 8, 'checkoutservice2-0': 9, 'checkoutservice-2': 10, 'checkoutservice-1': 11, 'paymentservice-0': 12, 'paymentservice2-0': 13, 'paymentservice-2': 14, 'paymentservice-1': 15, 'productcatalogservice-0': 16, 'productcatalogservice2-0': 17, 'productcatalogservice-2': 18, 'productcatalogservice-1': 19, 'shippingservice-0': 20, 'shippingservice2-0': 21, 'shippingservice-2': 22, 'shippingservice-1': 23, 'frontend-0': 24, 'frontend2-0': 25, 'frontend-2': 26, 'frontend-1': 27, 'recommendationservice-0': 28, 'recommendationservice2-0': 29, 'recommendationservice-2': 30, 'recommendationservice-1': 31, 'cartservice-0': 32, 'cartservice2-0': 33, 'cartservice-2': 34, 'cartservice-1': 35, 'currencyservice-0': 36, 'currencyservice2-0': 37, 'currencyservice-2': 38, 'currencyservice-1': 39, 'node-1': 40, 'node-2': 41, 'node-3': 42, 'node-4': 43, 'node-5': 44, 'node-6': 45}
        elif data_set == "D2":
            self.all_enum = {'adservice-0': 0, 'adservice-1': 1, 'adservice-2': 2,'aiops-k8s-01': 3, 'aiops-k8s-02': 4, 'aiops-k8s-03': 5, 'aiops-k8s-04': 6, 'aiops-k8s-05': 7, 'aiops-k8s-06': 8, 'aiops-k8s-07': 9, 'aiops-k8s-08': 10,'cartservice-0': 11, 'cartservice-1': 12, 'cartservice-2': 13,'checkoutservice-2': 14,'currencyservice-0': 15, 'currencyservice-1': 16, 'currencyservice-2': 17,'emailservice-0': 18, 'emailservice-1': 19, 'emailservice-2': 20,'frontend-0': 21, 'frontend-1': 22, 'frontend-2': 23,'k8s-master1': 24, 'k8s-master2': 25, 'k8s-master3': 26,'paymentservice-0': 27, 'paymentservice-1': 28, 'paymentservice-2': 29,'productcatalogservice-0': 30, 'productcatalogservice-1': 31, 'productcatalogservice-2': 32,'recommendationservice-0': 33,'redis-cart-0':34,'shippingservice-0': 35, 'shippingservice-1': 36, 'shippingservice-2': 37,'tidb_pd': 38, 'tidb_tidb': 39, 'tidb_tikv': 40}
           
       