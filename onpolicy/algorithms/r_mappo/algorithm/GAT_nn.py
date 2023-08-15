import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
# from torch_sparse import SparseTensor

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer,
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, device='cpu'):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征维度
        self.out_features = out_features   # 节点表示向量的输出特征维度
        self.dropout = dropout    # dropout参数
        self.alpha = alpha     # leakyrelu激活的参数
        self.concat = concat   # 如果为true, 再进行elu激活
        self.device = device
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), device=self.device))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1), device=self.device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化
        
        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.to(device)
    
    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]    # N 图的节点数
        
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e, device=self.device)    # 将没有连接的边置为负无穷
        attention = torch.where(adj>0, e, zero_vec)   # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)   # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    
    
    
class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads, node_num, n_thr, device='cpu'):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout 
        self.node_num = node_num
        self.node_num_batch = node_num * n_thr
        self.device = device
        
        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True, device=self.device) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)   # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout,alpha=alpha, concat=False, device=self.device)
    
    def to_adj(self, edge_index):
        # adj = torch.zeros(self.node_num_batch, self.node_num_batch, device=self.device) ## 其实是 node_num* n_thr
        adj = torch.zeros(edge_index.shape[0], edge_index.shape[0], device=self.device) ## 其实是 node_num* n_thr
        for i in range(edge_index.shape[0]):
            for j in edge_index[i]:
                if j != -1:
                    div_n = i // self.node_num
                    if div_n >= 1:
                        j_ind = j + self.node_num * div_n
                    else:
                        j_ind = j
                    adj[i][j_ind.long()] = 1
                    
        return adj.long()
    
    def forward(self, x, edge_index, backward=False):
        adj = self.to_adj(edge_index)
        # if not backward:
        #     adj = self.to_adj(edge_index)
        # else:
        #     adj = torch.eye(x.shape[0]).long().to(self.device)
        #     n = x.shape[0]
        #     m = n*n//2
        #     indices = torch.randperm(n * n)[:m]
        #     adj.view(-1)[indices] = 1
            
        x = F.dropout(x, self.dropout, training=self.training)   # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)   # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))   # 输出并激活
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定



# # 定义一个小图
# in_channels, hidden_channels, out_channels, dropout, alpha, heads, node_num = 16, 8, 32, 0.8, 0.2, 2, 8
# model = GAT(in_channels, hidden_channels, out_channels, dropout, alpha, heads, node_num)
# # edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 0, 5, 4]], dtype=torch.long)
# edge_index = torch.tensor([[ 3., -1.,  1.,  6.], [-1., -1.,  2.,  7.]])



# x = torch.randn(8, 16)  # 节点特征向量维度为 16
# out = model(x, edge_index)

# print(out.shape)
    