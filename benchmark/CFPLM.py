import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch.nn.init as init



class GraphConvolutionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convolution_network_first = gnn.GCNConv(in_channels, hidden_channels)
        self.convolution_network_second = gnn.GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        first = torch.relu(self.convolution_network_first(x, edge_index, edge_attr))
        return torch.relu(self.convolution_network_second(first, edge_index, edge_attr))


class Adapter(nn.Module):   
    def __init__(self, in_channels,hidden_channel, out_channel):
        super().__init__()
        self.lin1=nn.Linear(in_channels, hidden_channel)
        self.lin2=nn.Linear(hidden_channel,out_channel)

    def forward(self, x):
        first=self.lin1(x)
        second=self.lin2(first)
        return second

class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, head_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_q1 = nn.Linear(in_dim1, head_dim * num_heads)
        self.proj_k2 = nn.Linear(in_dim2, head_dim * num_heads)
        self.proj_v2 = nn.Linear(in_dim2, head_dim * num_heads)

        self.norm = nn.LayerNorm(in_dim1)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(head_dim * num_heads, in_dim1)

    def forward(self, x1, x2):
        residual = x1

        batch_size = x1.size(0)

        q = self.proj_q1(x1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k2(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v2(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

   
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1) 
        attn = self.dropout(attn) 

    
        output = (attn @ v).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(output)
       
        return self.norm(residual + output)

class CFPLM_Model(nn.Module):

    def __init__(self, data_dict, device, embedding_dimension=256):
        super().__init__()
        self.xr = data_dict['rna_vector']
        self.xp = data_dict['protein_vector']
        self.feature_rna_dim = self.xr.shape[1]
        self.feature_pro_dim = self.xp.shape[1]
        self.embedding_dimension = embedding_dimension
        self.num_rs = len(self.xr)
        self.num_ps = len(self.xp)
        self.interaction = data_dict['inter_train']
        self.rna_index = data_dict['rna_index']
        self.rna_val = data_dict['rna_val']
        self.protein_index = data_dict['protein_index']
        self.protein_val = data_dict['protein_val']
        self.rna_adapter = Adapter(self.feature_rna_dim,512, self.embedding_dimension)
        self.rna_inter_adapter = Adapter(self.feature_rna_dim,512, self.embedding_dimension)
        self.protein_inter_adapter = Adapter(self.feature_pro_dim,512,self.embedding_dimension)
        self.protein_adapter = Adapter(self.feature_pro_dim,512,self.embedding_dimension)
        nn.init.xavier_normal_(self.rna_adapter.lin1.weight)
        nn.init.xavier_normal_(self.rna_adapter.lin2.weight)
        nn.init.xavier_normal_(self.rna_inter_adapter.lin1.weight)
        nn.init.xavier_normal_(self.rna_inter_adapter.lin2.weight)
        nn.init.xavier_normal_(self.protein_inter_adapter.lin1.weight)
        nn.init.xavier_normal_(self.protein_inter_adapter.lin2.weight)
        nn.init.xavier_normal_(self.protein_adapter.lin1.weight)
        nn.init.xavier_normal_(self.protein_adapter.lin2.weight)
        self.rna_conv = GraphConvolutionNetwork(embedding_dimension, embedding_dimension, embedding_dimension)
        self.rna_protein_conv = GraphConvolutionNetwork(embedding_dimension, embedding_dimension, embedding_dimension)
        self.protein_conv = GraphConvolutionNetwork(embedding_dimension, embedding_dimension, embedding_dimension)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension * 2, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dimension, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss(reduction='mean')
        self.cross_attn1 = CrossAttention(embedding_dimension, embedding_dimension)
        self.cross_attn2 = CrossAttention(embedding_dimension, embedding_dimension)


    def forward(self, index, labels, type):
        rna_vector = self.rna_adapter(self.xr)
        protein_vector = self.protein_adapter(self.xp)
        rna_int_vector = self.rna_inter_adapter(self.xr)
        protein_int_vector = self.protein_inter_adapter(self.xp)
        rna_protein_int_vector = torch.cat([rna_int_vector, protein_int_vector])

        rna_vec_gcn = self.rna_conv(rna_vector, self.rna_index, self.rna_val)
        rna_protein_int_gcn = self.rna_protein_conv(rna_protein_int_vector, self.interaction, None)
        protein_vec_gcn = self.protein_conv(protein_vector, self.protein_index, self.protein_val)
        rna_int_vec_gcn, protein_int_vec_gcn = torch.split(rna_protein_int_gcn, split_size_or_sections=[self.num_rs, self.num_ps])

        rna_int_vec_gcn = rna_int_vec_gcn.unsqueeze(0)
        protein_int_vec_gcn = protein_int_vec_gcn.unsqueeze(0)
        protein_vec_gcn = protein_vec_gcn.unsqueeze(0)
        rna_vec_gcn = rna_vec_gcn.unsqueeze(0)
        rna_attention_inter = self.cross_attn1(rna_int_vec_gcn, protein_int_vec_gcn)
        protein_attention_inter = self.cross_attn1(protein_int_vec_gcn, rna_int_vec_gcn) 
        rna_attention_coll = self.cross_attn2(rna_vec_gcn, protein_vec_gcn)
        protein_attention_coll = self.cross_attn2(protein_vec_gcn, rna_vec_gcn)
        rna_attention_inter=rna_attention_inter.squeeze(0)
        rna_attention_coll=rna_attention_coll.squeeze(0)
        protein_attention_inter=protein_attention_inter.squeeze(0)
        protein_attention_coll=protein_attention_coll.squeeze(0)

        fused_rr_hr_rp_hr = rna_attention_inter + rna_attention_coll
        fused_pp_hp_rp_hp = protein_attention_inter + protein_attention_coll

        gather = torch.cat([fused_rr_hr_rp_hr, fused_pp_hp_rp_hp])
        rna = gather[index[:, 0]]
        protein = gather[index[:, 1]]
        gather_data = torch.cat([rna, protein], dim=1)
        out_pre = self.mlp(gather_data)
        out = torch.squeeze(out_pre)
        pred_loss = self.criterion(out, labels)
        loss = pred_loss

        return out, loss

