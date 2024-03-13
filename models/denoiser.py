import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn import setup_module
from models.gnn import MLP as GNNMLP
from models.attn import Multi_CrossAttention
import math
from models.utils import create_activation, NormLayer, create_norm, drop_edge
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils.sparse import dense_to_sparse,to_torch_coo_tensor
import dgl
import pdb
import warnings
warnings.filterwarnings("ignore")
# class TimeEmbedding(nn.Module):
#     # use sin and cos to embed time
#     def __init__(self, dim, max_period):
#         super(TimeEmbedding, self).__init__()
#         self.dim = dim
#         self.max_period = max_period
#         self.embed = nn.Linear(1, dim, bias=False)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, activation, dropout, norm=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        # self.activation = activation
        self.activation = create_activation(activation)
        self.dropout = dropout
        self.norm = create_norm(norm)(hidden_dim)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x

class gPool(nn.Module):
    """
    Our implementation of the pooling methodology described in the paper, above you can see the pseudocode
    """
    def __init__(self,in_dim,ratio):
        super().__init__()
        self.in_dim=in_dim
        self.ratio=ratio
        self.p=nn.Linear(in_dim,1)
        self.sigmoid=nn.Sigmoid()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self,xl,edge_index):
        
        #y is described as a learnable projection, hence we estimated a linear layer to be a suitable choice
        
        y=self.p(xl) #/torch.norm(self.p(xl)) #DO WE HAVE TO INSERT THE NORM?
        
        k=int(self.ratio*len(y))#Number of selected nodes, the k in TopkPooling
        
        topks, idx =torch.topk(y,k,dim=0) #The k elements with top scores, and their corresponding idx
        
        y_hat=self.sigmoid(topks)
        xl_hat=xl[idx,:].squeeze() #Feature matrix of topk nodes
        xl1=xl_hat * y_hat  #Gate operation
        
        al = torch.as_tensor(to_scipy_sparse_matrix(edge_index,num_nodes=len(y)).todense(), device = self.device)
        al1 = torch.index_select(torch.as_tensor(al),0,idx.squeeze()) #no direct indexing because of ram overloading
        al1=torch.index_select(al1,1,idx.squeeze())
        sparsel1=dense_to_sparse(al1)
        edge_index_pooled=torch.sparse_coo_tensor(sparsel1[0],sparsel1[1]).coalesce().indices() #Our model elaborates coo tensors
        return xl1, edge_index_pooled, idx.squeeze()
    
class gUnpool(nn.Module):
    """
    Our implementation of the unpooling methodology described in the paper
    """
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self,xl,idx,up_shape):
        up_nodes,C = up_shape #We extract the number of nodes and features the graph has to return to
        xl1=torch.zeros((up_nodes,xl.shape[1]), device = self.device)
        xl1[idx]=xl #We fetch the current feature matrix and sorround it of zeros to have the desired shape
        return xl1

class Denoiser(nn.Module):
    def __init__(self, gnn_type, 
                 in_dim, 
                 noise_emb_dim, 
                 hidden_dim, 
                 num_layers, 
                 activation, 
                 feat_drop, 
                 attn_drop, 
                 negative_slope, 
                 residual, 
                 norm, 
                 enc_nhead,
                 mlp_layers,):
        super(Denoiser, self).__init__()
        self.gnnlayers = setup_module(m_type=gnn_type,  
                                      enc_dec="encoding", 
                                      in_dim=(in_dim+noise_emb_dim),
                                      num_hidden=hidden_dim, 
                                      out_dim=hidden_dim, 
                                      num_layers=num_layers, 
                                      nhead=enc_nhead,
                                      nhead_out=enc_nhead,
                                      concat_out=True,
                                      activation=activation,
                                      dropout=feat_drop,
                                      attn_drop=attn_drop,
                                      negative_slope=negative_slope,
                                      residual=residual,
                                      norm=norm)
        if gnn_type == "gat":
            self.fc_in = hidden_dim*enc_nhead
        else:
            self.fc_in = hidden_dim
        self.fc = MLP(in_dim=self.fc_in, 
                         hidden_dim=hidden_dim, 
                         out_dim=in_dim, 
                         num_layers=mlp_layers, 
                         activation=activation, 
                         dropout=feat_drop)
        
        self.time_emb = SinusoidalPosEmb(noise_emb_dim)
        

    def forward(self, noised_data, t, g):
        # Concatenate t_emb with noised_data
        assert noised_data.shape[0] == t.shape[0] == 1
        assert len(noised_data.shape) == 3
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
            t = t.expand(-1, noised_data.shape[1])
        assert len(t.shape) == 2
        # same device
        assert noised_data.device == t.device 
        t_emb = self.time_emb(t)
        noised_data_t = torch.cat((noised_data, t_emb), dim=-1)
        # Encoding
        noised_data_t = noised_data_t.squeeze(0)
        h, _ = self.gnnlayers(g, noised_data_t, return_hidden=True)
        h = h.unsqueeze(0)
        # Decoding
        h = self.fc(h)
        return h
    
    

class DenoiserMLP(nn.Module):
    def __init__(self, gnn_type, 
                 in_dim, 
                 noise_emb_dim, 
                 hidden_dim, 
                 num_layers, 
                 activation, 
                 feat_drop, 
                 attn_drop, 
                 negative_slope, 
                 residual, 
                 norm, 
                 enc_nhead,
                 mlp_layers,):
        super(DenoiserMLP, self).__init__()
        self.MLP = MLP(in_dim=(in_dim+noise_emb_dim),
                         hidden_dim=hidden_dim,
                         out_dim=in_dim,
                         num_layers=num_layers,
                         activation=activation,
                         dropout=feat_drop)
        
        self.fc = MLP(in_dim=in_dim, 
                         hidden_dim=hidden_dim, 
                         out_dim=in_dim, 
                         num_layers=mlp_layers, 
                         activation=activation, 
                         dropout=feat_drop)
        
        self.time_emb = SinusoidalPosEmb(noise_emb_dim)
        
        
    def forward(self, noised_data, t, g):
        # Concatenate t_emb with noised_data
        assert noised_data.shape[0] == t.shape[0] == 1
        assert len(noised_data.shape) == 3
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
            t = t.expand(-1, noised_data.shape[1])
        assert len(t.shape) == 2
        # same device
        assert noised_data.device == t.device 
        t_emb = self.time_emb(t)
        noised_data_t = torch.cat((noised_data, t_emb), dim=-1)
        # Decoding
        h_res = self.MLP(noised_data_t)
        h = h_res + self.fc(noised_data)
        return h
        
class DenoiserUnet(nn.Module):
    def __init__(self, gnn_type, 
                 in_dim, 
                 noise_emb_dim, 
                 hidden_dim, 
                 num_layers, 
                 activation, 
                 feat_drop, 
                 attn_drop, 
                 negative_slope, 
                 residual, 
                 norm, 
                 enc_nhead,
                 use_timeembedding=True,):
        super(DenoiserUnet, self).__init__()
        # self.gnnlayers = setup_module(m_type=gnn_type,  
        #                               enc_dec="encoding", 
        #                               in_dim=(in_dim+noise_emb_dim),
        #                               num_hidden=hidden_dim, 
        #                               out_dim=hidden_dim, 
        #                               num_layers=num_layers, 
        #                               nhead=enc_nhead,
        #                               nhead_out=enc_nhead,
        #                               concat_out=True,
        #                               activation=activation,
        #                               dropout=feat_drop,
        #                               attn_drop=attn_drop,
        #                               negative_slope=negative_slope,
        #                               residual=residual,
        #                               norm=norm)
        assert num_layers%2 == 1
        self.gpoollayers = nn.ModuleList()
        self.unpoollayers = nn.ModuleList()
        for i in range((num_layers-1)//2):
            self.gpoollayers.append(gPool(hidden_dim, 0.5))
        
        for i in range((num_layers-1)//2):
            self.unpoollayers.append(gUnpool(hidden_dim, hidden_dim))
        
        
        self.in_map = nn.Linear(in_dim, hidden_dim)
        self.gnnlayers = nn.ModuleList()
        for i in range(num_layers):
            if i>0 or (not use_timeembedding):
                self.gnnlayers.append(setup_module(m_type=gnn_type,  
                                      enc_dec="encoding", 
                                      in_dim=hidden_dim,
                                      num_hidden=hidden_dim, 
                                      out_dim=hidden_dim, 
                                      num_layers=1, 
                                      nhead=enc_nhead,
                                      nhead_out=enc_nhead,
                                      concat_out=True,
                                      activation=activation,
                                      dropout=feat_drop,
                                      attn_drop=attn_drop,
                                      negative_slope=negative_slope,
                                      residual=residual,
                                      norm=norm))
            elif i==0 and use_timeembedding:
                self.gnnlayers.append(setup_module(m_type=gnn_type,  
                                      enc_dec="encoding", 
                                      in_dim=(hidden_dim+noise_emb_dim),
                                      num_hidden=hidden_dim, 
                                      out_dim=hidden_dim, 
                                      num_layers=1, 
                                      nhead=enc_nhead,
                                      nhead_out=enc_nhead,
                                      concat_out=True,
                                      activation=activation,
                                      dropout=feat_drop,
                                      attn_drop=attn_drop,
                                      negative_slope=negative_slope,
                                      residual=residual,
                                      norm=norm))
        
        self.fc = nn.Linear(hidden_dim, in_dim)
        self.use_timeembedding = use_timeembedding
        if use_timeembedding:
            self.time_emb = SinusoidalPosEmb(noise_emb_dim)
        # self.time_emb = SinusoidalPosEmb(noise_emb_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        

    def forward(self, noised_data, t, g):
        # Concatenate t_emb with noised_data
        assert noised_data.shape[0] == t.shape[0] == 1
        assert len(noised_data.shape) == 3
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
            t = t.expand(-1, noised_data.shape[1])
        assert len(t.shape) == 2
        noised_data = self.in_map(noised_data)
        if self.use_timeembedding:
            t_emb = self.time_emb(t)
            # input mapping
            noised_data_t = torch.cat((noised_data, t_emb), dim=-1)
        else:
            noised_data_t = noised_data
        # Encoding
        noised_data_t = noised_data_t.squeeze(0) # N x F
        noised_data_pools = []
        idxs = []
        gs = []
        for i in range((len(self.gpoollayers))):
            noised_data_t, _ = self.gnnlayers[i](g, noised_data_t, return_hidden=True)
            noised_data_pools.append(noised_data_t)
            gs.append(g)
            edge_index = torch.cat([x.unsqueeze(0) for x in g.all_edges()], dim=0)
            noised_data_t, edge_index, idx = self.gpoollayers[i](noised_data_t, edge_index)
            # build graph
            # try:
            g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=noised_data_t.shape[0])
            g = dgl.add_self_loop(g)
            # except:
                # pdb.set_trace()
            idxs.append(idx)
        noised_data_t, _ = self.gnnlayers[len(self.gpoollayers)](g, noised_data_t, return_hidden=True)

        #Decoding
        for i in range((len(self.unpoollayers))):
            noised_data_t = self.unpoollayers[i](noised_data_t, idxs[-i-1], noised_data_pools[-i-1].shape)
            noised_data_t = noised_data_t + noised_data_pools[-i-1] # could be concat
            noised_data_t, _ = self.gnnlayers[i+len(self.gpoollayers)+1](gs[-i-1], noised_data_t, return_hidden=True)
        # Readout
        # layer norm
        noised_data_t = self.norm(noised_data_t)
        h = self.fc(noised_data_t.unsqueeze(0))
        # h, _ = self.gnnlayers(g, noised_data_t, return_hidden=True)
        # h = h.unsqueeze(0)
        # # Decoding
        # h = self.fc(h)
        return h
    
class DeepLPSI(nn.Module):
    def __init__(self, self_loop):
        super(DeepLPSI, self).__init__()
        self.gnn = setup_module(m_type="gcn",
                                enc_dec="decoding", 
                                in_dim=1,
                                num_hidden=8, 
                                out_dim=1, 
                                num_layers=2, 
                                nhead=1,
                                nhead_out=1,
                                concat_out=True,
                                activation="prelu",
                                dropout=0.5,
                                attn_drop=0.5,
                                negative_slope=0.2,
                                residual=True,
                                norm='layernorm')
        self.self_loop = self_loop
        
    def forward(self, y, g):
        assert len(y.shape) == 2
        y[y==0]=-1
        for i in range(self.self_loop):
            y = y + self.gnn(g, y)

        return y
    
    def condition(self, y, g):
        adj = self.draw_adj(g)
        assert len(y.shape) == 2
        assert len(adj.shape) == 2
        assert y.shape[0] == adj.shape[0] == adj.shape[1]
        laplacian = torch.diag(torch.sum(adj, dim=1)) - adj
        y_init = y.clone()
        y = self(y, g)
        # y = y_init * y
        y = torch.matmul(laplacian, y)
        return y
    
    @staticmethod
    def normalize_adj(adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        """Symmetrically normalize adjacency matrix."""
        # adj = sp.coo_matrix(adj)
        rowsum = torch.tensor(adj.sum(1))
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized =  adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)
        return adj_normalized
    
    @staticmethod
    def LPSI_coverage(adj, adj_norm, Y, alpha, atol=1e-6):
        matrix_i = torch.eye(adj.shape[1]).to(adj.device)

        adj_normalized = adj_norm
        matrix_s = adj_normalized
        y = Y.reshape((-1, 1)).clone()
        y[y==0]=-1.
        _mtrx = torch.tensor(matrix_i - alpha * matrix_s).to(y.device)
        t_mtrx = torch.inverse(_mtrx)
        _converge = (1 - alpha) * t_mtrx
        converge = torch.matmul(_converge, y).flatten()
        source_set = torch.zeros_like(Y).to(Y.device)

        # Step 6: For each original infected node i
        for i in range(len(Y)):
            if Y[i] > 0:  # Check if node i is an original infected node
                # If G*_i is greater than all of i's neighbors' G* values
                if all(converge[i] > converge[j] for j in range(len(converge)) if adj[i, j] == 1):
                    # Add i to S
                    source_set[i] = 1
        
        return source_set, 0, converge
    
    @staticmethod
    def draw_adj(g):
        edge_index = g.all_edges()
        num_nodes = g.num_nodes()
        adj = torch.zeros((num_nodes, num_nodes)).to(edge_index[0].device)
        for i in range(edge_index[0].shape[0]):
            adj[edge_index[0][i], edge_index[1][i]] = 1
            adj[edge_index[1][i], edge_index[0][i]] = 1
        return adj

class deepLPSI_ablation(nn.Module):
    # use identity
    def __init__(self, self_loop):
        super(deepLPSI_ablation, self).__init__()
        self.self_loop = self_loop
        self.mlp = MLP(in_dim=1,
                          hidden_dim=8,
                          out_dim=1,
                          num_layers=2,
                          activation="prelu",
                          dropout=0.5,
                          norm="layernorm")
    def forward(self, y, g):
        return self.mlp(y)
    def condition(self, y, g):
        return self(y, g)
    
    @staticmethod
    def normalize_adj(adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        """Symmetrically normalize adjacency matrix."""
        # adj = sp.coo_matrix(adj)
        rowsum = torch.tensor(adj.sum(1))
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized =  adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)
        return adj_normalized
    
    @staticmethod
    def LPSI_coverage(adj, adj_norm, Y, alpha, atol=1e-6):
        matrix_i = torch.eye(adj.shape[1]).to(adj.device)

        adj_normalized = adj_norm
        matrix_s = adj_normalized
        y = Y.reshape((-1, 1)).clone()
        y[y==0]=-1.
        _mtrx = torch.tensor(matrix_i - alpha * matrix_s).to(y.device)
        t_mtrx = torch.inverse(_mtrx)
        _converge = (1 - alpha) * t_mtrx
        converge = torch.matmul(_converge, y).flatten()
        source_set = torch.zeros_like(Y).to(Y.device)

        # Step 6: For each original infected node i
        for i in range(len(Y)):
            if Y[i] > 0:  # Check if node i is an original infected node
                # If G*_i is greater than all of i's neighbors' G* values
                if all(converge[i] > converge[j] for j in range(len(converge)) if adj[i, j] == 1):
                    # Add i to S
                    source_set[i] = 1
        
        return source_set, 0, converge
    
    @staticmethod
    def draw_adj(g):
        edge_index = g.all_edges()
        num_nodes = g.num_nodes()
        adj = torch.zeros((num_nodes, num_nodes)).to(edge_index[0].device)
        for i in range(edge_index[0].shape[0]):
            adj[edge_index[0][i], edge_index[1][i]] = 1
            adj[edge_index[1][i], edge_index[0][i]] = 1
        return adj

class DeepLPSI2(nn.Module):
    def __init__(self, self_loop):
        super(DeepLPSI2, self).__init__()
        self.gnn = setup_module(m_type="gcn",
                                enc_dec="decoding", 
                                in_dim=8,
                                num_hidden=8, 
                                out_dim=8, 
                                num_layers=self_loop, 
                                nhead=1,
                                nhead_out=1,
                                concat_out=True,
                                activation="prelu",
                                dropout=0.5,
                                attn_drop=0.5,
                                negative_slope=0.2,
                                residual=True,
                                norm='layernorm')
        # self.self_loop = self_loop
        self.input_proj = nn.Linear(1, 8)
        self.output_proj = nn.Linear(8, 128)
        
    def forward(self, y, g):
        assert len(y.shape) == 2
        y = self.input_proj(y)
        # for i in range(self.self_loop):
        y = self.gnn(g, y)

        return y
    
    def condition(self, y, g):
        adj = self.draw_adj(g)
        assert len(y.shape) == 2
        assert len(adj.shape) == 2
        assert y.shape[0] == adj.shape[0] == adj.shape[1]
        laplacian = torch.diag(torch.sum(adj, dim=1)) - adj
        y_init = y.clone()
        y = self(y, g)
        # y = y_init * y
        y = torch.matmul(laplacian, y)
        y = self.output_proj(y)
        return y
    
    @staticmethod
    def normalize_adj(adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        """Symmetrically normalize adjacency matrix."""
        # adj = sp.coo_matrix(adj)
        rowsum = torch.tensor(adj.sum(1))
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_normalized =  adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)
        return adj_normalized
    
    @staticmethod
    def LPSI_coverage(adj, adj_norm, Y, alpha, atol=1e-6):
        matrix_i = torch.eye(adj.shape[1]).to(adj.device)

        adj_normalized = adj_norm
        matrix_s = adj_normalized
        y = Y.reshape((-1, 1)).clone()
        y[y==0]=-1.
        _mtrx = torch.tensor(matrix_i - alpha * matrix_s).to(y.device)
        t_mtrx = torch.inverse(_mtrx)
        _converge = (1 - alpha) * t_mtrx
        converge = torch.matmul(_converge, y).flatten()
        source_set = torch.zeros_like(Y).to(Y.device)

        # Step 6: For each original infected node i
        # for i in range(len(Y)):
            # if Y[i] > 0:  # Check if node i is an original infected node
            #     # If G*_i is greater than all of i's neighbors' G* values
            #     if all(converge[i] > converge[j] for j in range(len(converge)) if adj[i, j] == 1):
            #         # Add i to S
            #         source_set[i] = 1
        n = Y.size(0)
        for i in range(n):
            if Y[i] > 0:  # 检查节点 i 是否是原始感染节点
                # 获取节点 i 的所有邻居节点
                neighbors = torch.where(adj[i] == 1)[0]
                # 检查 G*_i 是否大于所有邻居节点的 G* 值
                if torch.all(converge[i] > converge[neighbors]):
                    # 将 i 添加到 S 中
                    source_set[i] = 1
        return source_set, 0, converge
    
    @staticmethod
    def draw_adj(g):
        edge_index = g.all_edges()
        num_nodes = g.num_nodes()
        adj = torch.zeros((num_nodes, num_nodes)).to(edge_index[0].device)
        for i in range(edge_index[0].shape[0]):
            adj[edge_index[0][i], edge_index[1][i]] = 1
            adj[edge_index[1][i], edge_index[0][i]] = 1
        return adj

class DenoiseAdvisor(nn.Module):
    def __init__(self, gnn_type, 
                 in_dim, 
                 noise_emb_dim, 
                 hidden_dim, 
                 num_advisors,
                 num_layers, 
                 activation, 
                 feat_drop, 
                 attn_drop, 
                 negative_slope, 
                 residual, 
                 norm, 
                 enc_nhead,
                 mlp_layers,):
        super(DenoiseAdvisor, self).__init__()
        self.input_linear = nn.Linear(in_dim, hidden_dim)
        self.time_emb = SinusoidalPosEmb(noise_emb_dim)
        assert noise_emb_dim == hidden_dim
        if num_advisors>0:
            # self.cross_attn = Multi_CrossAttention(hidden_size_q=34, hidden_size_kv=34, all_head_size=enc_nhead * 34, head_num=enc_nhead)
            self.cross_attn = setup_module(m_type='gat',
                                        enc_dec="encoding", 
                                        in_dim=1,
                                        num_hidden=hidden_dim, 
                                        out_dim=hidden_dim//4, 
                                        num_layers=1, 
                                        nhead=4,
                                        nhead_out=4,
                                        concat_out=True,
                                        activation=activation,
                                        dropout=feat_drop,
                                        attn_drop=attn_drop,
                                        negative_slope=negative_slope,
                                        residual=residual,
                                        norm=norm)
        else:
            self.cross_attn = None
        self.gencoder = setup_module(m_type=gnn_type,  
                                      enc_dec="encoding", 
                                      in_dim=(hidden_dim),
                                      num_hidden=hidden_dim, 
                                      out_dim=hidden_dim, 
                                      num_layers=num_layers, 
                                      nhead=4,
                                      nhead_out=4,
                                      concat_out=True,
                                      activation=activation,
                                      dropout=feat_drop,
                                      attn_drop=attn_drop,
                                      negative_slope=negative_slope,
                                      residual=residual,
                                      norm=norm)
        self.gencoder_norm = nn.Sequential(nn.LayerNorm(hidden_dim))
        self.gdecoder = GNNMLP(num_layers=mlp_layers, 
                               input_dim=hidden_dim, 
                               hidden_dim=hidden_dim, 
                               output_dim=in_dim, 
                               activation=activation, 
                               norm="layernorm",)
        
        self.conditional = DeepLPSI(self_loop=2)
        # self.conditional = deepLPSI_ablation(self_loop=2)
    
    def conditioning(self, y, g):
        return self.conditional.condition(y, g)
        
    def forward(self, noised_data, t, g, cond, advisors=None):
        # Concatenate t_emb with noised_data
        assert noised_data.shape[0] == t.shape[0] == 1
        assert len(noised_data.shape) == 3
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
            t = t.expand(-1, noised_data.shape[1])
        assert len(t.shape) == 2
        # same device
        assert noised_data.device == t.device 
        t_emb = self.time_emb(t)
        
        # if self.cross_attn is not None:
            # advisors = advisors.unsqueeze(0)
            # advisors = advisors.transpose(1,2)
            # noised_data = noised_data.transpose(1,2)
            # assert advisors.shape[-1] == noised_data.shape[-1]
            # advisors = advisors.squeeze(-1).transpose(0,1) # N x (F x H)
            # noised_data = noised_data + self.cross_attn(noised_data, advisors)
            # noised_data = noised_data.transpose(1,2)
            
        
        noised_data_t = self.input_linear(noised_data)
        
        noised_data_t = noised_data_t + self.cross_attn(g,advisors)
        
        noised_data_t = noised_data_t + t_emb
        
        # attention
        # noised_data_t = noised_data_t # N x F
        
        
        noised_data_t = noised_data_t.squeeze(0)
        # Encoding
        # noised_data_t = noised_data_t.squeeze(0)
        h = self.gencoder(g, noised_data_t)
        h = self.gencoder_norm(h)
        h = h + noised_data_t
        # conditioning
        # cond = self.conditional.condition(y, g)
        h = h + cond # multiply or add
        # h = h + cond # multiply or add
        
        # Decoding
        h = self.gdecoder(h)
        return h.unsqueeze(0)
    
