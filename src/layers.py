import torch
import dgl
import math
import numpy as np

class TimeEncode(torch.nn.Module):
    # dim is the dimension of time encoding
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)# input(each scalar time) (batch_size, 1) and output (batch_size, dim); input.W^T + b;
        # note that (1, dim) is in fact the shape of W^T :)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))# shape = (dim, 1) internally; more generally (out_features×in_features)
        self.w.bias = torch.nn.Parameter(torch.zeros(dim))# shape = (dim,) but it acts like [batch, dim] after broadcasting.

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))# Ensure input is of shape (batch_size, 1)
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
        # h is a big tensor containing all node embeddings 
        # for a batch of edges, h.shape = [total_rows, dim_in]
        # each row = one node embedding vector (shape [dim_in])
        # for each edge in the batch, you have:
        # 1 (src) + 1 (pos_dst) + neg_samples (neg_dst) = (neg_samples + 2) embeddings
        # h.shape[0] = total number of node embeddings = num_edges × (2 + neg_samples)
        num_edge = h.shape[0] // (neg_samples + 2)# how many original edges (source–positive_dst pairs) are in the batch
        h_src = self.src_fc(h[:num_edge])# First segment = source nodes; sclicing row equivalent to h[0:num_edge, :] 
                                        # shape [num_edges, dim_in]
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])# we only sample negative dst, good to know; source nodes remain the same
                                                 # shape [num_edges * neg_samples, dim_in]
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)#[num_edges, dim_in]
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)#.tile(neg_samples, 1) repeats existing embedding along the batch dimension to get shape [num_edges * neg_samples, dim_in]
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)#[num_edges, 1], [num_edges×neg_samples, 1]; xW+b


class TransfomerAttentionLayer(torch.nn.Module):

    def __init__(self, dim_node_feat, dim_edge_feat, dim_time, num_head, dropout, att_dropout, dim_out, combined=False, is_hetero=False, num_relations=1):
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
        self.is_hetero = is_hetero
        self.num_relations = num_relations

        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)

        if is_hetero:
            self.rel_emb = torch.nn.Embedding(num_relations + 1, dim_out)# +1 for padding index

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
            if is_hetero:
                self.w_k_r = torch.nn.Linear(dim_out, dim_out)
                self.w_v_r = torch.nn.Linear(dim_out, dim_out)
        else:
            if dim_node_feat + dim_time > 0:
                self.w_q = torch.nn.Linear(dim_node_feat + dim_time, dim_out)
            eff_edge_dim = dim_edge_feat + (dim_out if is_hetero else 0)
            self.w_k = torch.nn.Linear(dim_node_feat + eff_edge_dim + dim_time, dim_out)
            self.w_v = torch.nn.Linear(dim_node_feat + eff_edge_dim + dim_time, dim_out)

        self.w_out = torch.nn.Linear(dim_node_feat + dim_out, dim_out)#you concatenate the original node features (residual connection)
        self.layer_norm = torch.nn.LayerNorm(dim_out)# normalize the output embeddings

    def forward(self, b):
        # b is a DGL block representing the sampled neighborhood for the current batch
        # Remember source = message sender (neighbor), destination = receiver (center node)
        # b.srcdata['h'] source node embeddings (includes dst + sampled neighbors)
        # holds also the input node features for every source node before forward pass [num_src_nodes, dim_node_feat]
        # b.dstdata['h'] destination node whose embeddings we want to compute
        # b.edata['f']	edge features
        # b.edata['dt']	time differences between source & destination shape is [num_edges]
        # b.edges()	returns (src_idx, dst_idx) pairs for all edges
        # b.num_edges()	number of temporal edges
        # b.num_dst_nodes()	number of destination nodes (to update)
        assert(self.dim_time + self.dim_node_feat + self.dim_edge_feat > 0 or self.is_hetero)
        # device = b.srcdata['h'].device might be useful
        if b.num_edges() == 0:
            return torch.zeros((b.num_dst_nodes(), self.dim_out), device=torch.device('cuda:0'))
        if self.is_hetero:
            rel_vec = self.rel_emb(b.edata['rel_type'].long()) # [num_edges, dim_out]
        else:
            rel_vec = None
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
                K += self.w_k_t(time_feat)# it’s the listener that aggregates past messages
                V += self.w_v_t(time_feat)#reason why there are no expansion here like in Q
            if self.is_hetero:
                K += self.w_k_r(rel_vec)
                V += self.w_v_r(rel_vec)
            Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))#[num_edges, num_head, dim_per_head]
            K = torch.reshape(K, (K.shape[0], self.num_head, -1))#dim_per_head = dim_out / num_head
            V = torch.reshape(V, (V.shape[0], self.num_head, -1))
            att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
            # Q * K [num_edges, num_head, dim_per_head]
            # after sum along dim=2, shape = [num_edges, num_head]
            #DGL uses b to determine which edges belong to the same destination node,
            #so that it can apply softmax normalization only within that node's incoming edges.
            att = self.att_dropout(att)# randomly drops out some attention weights
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))
            # att[:, :, None] expands attention shape to [num_edges, num_head, 1]
            # after reshape back to [num_edges, dim_out]
            b.edata['v'] = V #This attaches the weighted edge features to the graph as edge data
            b.update_all(dgl.function.copy_edge('v', 'm'), dgl.function.sum('m', 'h'))
            # For each destination node 'v', copy the edge feature 'v' -> 'm'
            # Then sum all 'm' from incoming edges into the destination's node data 'h'
        else:#are no longer applying w_k or w_v to pure node-level
            if self.is_hetero:
                # edge_input = [edge_feat, rel_vec] or  [rel_vec]
                if self.dim_edge_feat > 0:
                    assert 'f' in b.edata, "dim_edge_feat > 0 but no 'f' in b.edata"
                    edge_input = torch.cat([b.edata['f'], rel_vec], dim=1)
                else:
                    edge_input = rel_vec

                # Q from dest nodes (+ time if available)
                if self.dim_time > 0:
                    Q_in = torch.cat(
                        [b.srcdata['h'][:b.num_dst_nodes()], zero_time_feat], dim=1
                    )
                else:
                    Q_in = b.srcdata['h'][:b.num_dst_nodes()]

                Q = self.w_q(Q_in)[b.edges()[1]]

                # K,V from source-only nodes + edge_input (+ time)
                if self.dim_time > 0:
                    KV_in = torch.cat(
                        [b.srcdata['h'][b.num_dst_nodes():], edge_input, time_feat],
                        dim=1 )
                else:
                    KV_in = torch.cat(
                        [b.srcdata['h'][b.num_dst_nodes():], edge_input],
                        dim=1)
                K = self.w_k(KV_in)
                V = self.w_v(KV_in)
            else:
                if self.dim_time == 0 and self.dim_node_feat == 0:#no node context
                    Q = torch.ones((b.num_edges(), self.dim_out), device=torch.device('cuda:0'))#Q = dummy constant vector 
                    #(no node info, so just neutral query)
                    K = self.w_k(b.edata['f'])#linear projections of edge features only
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
            V = torch.reshape(V*att[:, :, None], (V.shape[0], -1)) # shape back to [num_edges, dim_out]
            b.srcdata['v'] = torch.cat([torch.zeros((b.num_dst_nodes(), V.shape[1]), device=torch.device('cuda:0')), V], dim=0)   
            # DGL expects 'v' to be defined for every node in b.srcdata
            # i.e shape (num_src_nodes, dim_out) including both destinations and source-only nodes
            #        
            b.update_all(dgl.function.copy_u('v', 'm'), dgl.function.sum('m', 'h'))
        if self.dim_node_feat != 0:
            # b.dstdata['h'] now stores the aggregated message for each destination node
            # b.srcdata['h'] original input embeddings
            rst = torch.cat([b.dstdata['h'], b.srcdata['h'][:b.num_dst_nodes()]], dim=1)
            # shape (num_dst_nodes, dim_out)
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

            