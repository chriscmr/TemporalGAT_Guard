import torch
import dgl
import math
import numpy as np

class TimeEncode(torch.nn.Module):
    #dim is the dimension of time encoding
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)#input(each scalar time) (batch_size, 1) and output (batch_size, dim); input.W^T + b;
        #note that (1, dim) is in fact the shape of W^T :)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))#shape = (dim, 1); more generally (out_features×in_features)
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))# shape = (dim,) but it acts like [batch, dim] after broadcasting.

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))## Ensure input is of shape (batch_size, 1)
        return output

class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        #dim_in: the dimensionality of the node embeddings
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)#transforms the source node embeddings; 
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)#transforms the destination node embeddings
        self.out_fc = torch.nn.Linear(dim_in, 1)#maps the final combined edge representation to a scalar score before sigmoid

    def forward(self, h, neg_samples=1):
        #h is a big tensor containing all node embeddings 
        #for a batch of edges, h.shape = [total_rows, dim_in]
        #each row = one node embedding vector (shape [dim_in])
        #for each edge in the batch, you have:
        #1 (src) + 1 (pos_dst) + neg_samples (neg_dst) = (neg_samples + 2) embeddings
        #h.shape[0] = total number of node embeddings = num_edges × (2 + neg_samples)
        num_edge = h.shape[0] // (neg_samples + 2)#how many original edges (source–positive_dst pairs) are in the batch
        h_src = self.src_fc(h[:num_edge])#First segment = source nodes; sclicing row equivalent to h[0:num_edge, :] 
                                        #shape [num_edges, dim_in]
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])#we only sample negative dst, good to know; source nodes remain the same
                                                 #shape [num_edges * neg_samples, dim_in]
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)#[num_edges, dim_in]
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)#.tile(neg_samples, 1) repeats existing embedding along the batch dimension to get shape [num_edges * neg_samples, dim_in]
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)#[num_edges, 1], [num_edges×neg_samples, 1]; xW+b


class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False):
        super(TransfomerAttentionLayer, self).__init__()
        self.num_head = num_head
        self.dim_node_feat = dim_node_feat# dimension of node features
        self.dim_edge_feat = dim_edge_feat# dimension of edge features
        self.dim_time = dim_time# dimension of time encoding
        self.dim_out = dim_out# dimension of output embedding per head
        self.dropout = torch.nn.Dropout(dropout)#on the values
        self.att_dropout = torch.nn.Dropout(att_dropout)#on the attention coefficients
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.combined = combined# whether to learn separate Q,K,V projections per feature type (node/edge/time) or combine all into one vector first
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combined:#Each feature type (node, edge, time) gets its own projection.
            if dim_node_feat > 0:# Node parts
                self.w_q_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_k_n = torch.nn.Linear(dim_node_feat, dim_out)
                self.w_v_n = torch.nn.Linear(dim_node_feat, dim_out)
            if dim_edge_feat > 0:# Edge parts
                self.w_k_e = torch.nn.Linear(dim_edge_feat, dim_out)
                self.w_v_e = torch.nn.Linear(dim_edge_feat, dim_out)
            if dim_time > 0:# Time parts
                self.w_q_t = torch.nn.Linear(dim_time, dim_out)
                self.w_k_t = torch.nn.Linear(dim_time, dim_out)
                self.w_v_t = torch.nn.Linear(dim_time, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            self.w_k = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + dim_edge_feat + dim_time, dim_out)
        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)#you concatenate the original node features (residual connection)
        self.layer_norm = torch.nn.LayerNorm(dim_out)# normalize the output embeddings

    def forward(self, b):
        # b is a DGL block epresenting the sampled neighborhood for the current batch
        # Remember source = message sender (neighbor), destination = receiver (center node) (yes confusing I know)
        # b.srcdata['h'] source node embeddings (includes dst + sampled neighbors)
        # holds the input node features for every source node before forward pass [num_src_nodes, dim_node_feat]
        # b.dstdata['h'] destination node whose embeddings we want to compute
        # b.edata['f']	edge features
        # b.edata['dt']	time differences between source & destination
        # b.edges()	returns (src_idx, dst_idx) pairs for all edges
        # b.num_edges()	number of temporal edges
        # b.num_dst_nodes()	number of destination nodes (to update)
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0)
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
        if self.dim_time > 0:
            time_feat = self.time_enc(b.edata['dt'])#edge gets a temporal encoding of its time difference Δt
            zero_time_feat = self.time_enc(torch.zeros(b.num_dst_nodes(), dtype=torch.float32, device=torch.device('cuda:0')))
        if self.combined:
            Q = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            K = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            V = torch.zeros((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
            if self.dim_node_feat > 0:
                Q += self.w_q_n(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                # b.srcdata['h'][:b.num_dst_nodes()] are the dest nodes [num_dst_nodes, dim_node_feat]
                # output shape after w_q_n [num_dst_nodes, dim_out]
                # b.edges()[1] destination indices of shape [num_edges]
                # It's an expension operation; we select rows according to dst indices (try on your own)
                # so shape after indexing = [num_edges, dim_out]
                K += self.w_k_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()]
                # neighbor (source-only) nodes; But the neighbor embeddings are
                # stored starting at index num_dst_nodes inside b.srcdata['h'], so we subtract
                # to index into src-only block
                V += self.w_v_n(b.srcdata['h'][b.num_dst_nodes():])[b.edges()[0] - b.num_dst_nodes()] 
            if self.dim_edge_feat > 0:
                K += self.w_k_e(b.edata['f'])
                V += self.w_v_e(b.edata['f'])
            if self.dim_time > 0:
                Q += self.w_q_t(zero_time_feat)[b.edges()[1]]
                K += self.w_k_t(time_feat)
                V += self.w_v_t(time_feat)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.edata['v'] = V
            b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
        else:
            if self.dim_time == 0 and self.dim_node_feat == 0:
                Q = torch.ones((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))
                K = self.w_k(b.edata['f'])
                V = self.w_v(b.edata['f'])
            elif self.dim_time == 0 and self.dim_edge_feat == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(b.srcdata['h'][b.num_dst_nodes():])
                V = self.w_v(b.srcdata['h'][b.num_dst_nodes():])
            elif self.dim_time == 0:
                Q = self.w_q(b.srcdata['h'][:b.num_dst_nodes()])[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f']], dim=1))
            elif self.dim_node_feat == 0:
                Q = self.w_q(zero_time_feat)[b.edges()[1]]
                K = self.w_k(torch.cat([b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.edata['f'], time_feat], dim=1))
            elif self.dim_edge_feat == 0:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], time_feat], dim=1))
            else:
                Q = self.w_q(torch.cat([b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1))[b.edges()[1]]
                K = self.w_k(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
                V = self.w_v(torch.cat([b.srcdata['h'][b.num_dst_nodes():], b.edata['f'], time_feat], dim=1))
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            att = self.att_dropout(att)
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)
            b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        if self.dim_node_feat != 0:
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
        else:
            rst = b.dstdata['h']
        rst = self.w_out(rst)
        rst = torch.nn.functional.relu(self.dropout(rst))
        return self.layer_norm(rst)

class IdentityNormLayer(torch.nn.Module):

    def __init__(self, dim_out):
        super(IdentityNormLayer, self).__init__()
        self.norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b):
        return self.norm(b.srcdata['h'])

            