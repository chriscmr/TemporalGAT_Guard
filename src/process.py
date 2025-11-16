import json
import numpy as np
import pandas as pd
import pickle
import torch
import os

def preprocess(data_name):
    # each line of the dataset looks like this
    # u_id, i_id, timestamp, label, f1, f2, f3, ..., f_d
    # user(source node), item(destination node), time when the interaction occurred
    # features → edge-specific features
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = [] 
    
    with open(data_name) as f:
        s = next(f)
        print(s) # Skips the first line (header)
        for idx, line in enumerate(f): 
            e = line.strip().split(',') #Splits each line into a list of strings
            u = int(e[0]) # user ID
            i = int(e[1]) # item ID     
            
            ts = float(e[2]) # timestamp
            label = int(e[3]) # label: 0/1 for negative/positive interaction
            
            feat = np.array([float(x) for x in e[4:]]) # edge features
            
            # Each interaction (edge) is one entry in these lists
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)



def reindex(df):
    # no missing integers
    assert(df.u.max() - df.u.min() + 1 == len(df.u.unique())) # n-p+1 :) number of element between max and min
    assert(df.i.max() - df.i.min() + 1 == len(df.i.unique()))
    
    # If users from 0..U-1
    # Then items start from U..U+I-1
    # ensure no overlap in node IDs
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u
    
    new_df = df.copy()
    print(new_df.u.max())
    print(new_df.i.max())
    
    new_df.i = new_i
    new_df.u += 1 # users \in [1, U]
    new_df.i += 1 # items \in [U+1, U+I]
    new_df.idx += 1 # edge idx \in [1, E]
    # total nodes = U + I
    
    print(new_df.u.max())
    print(new_df.i.max())
    
    return new_df



def run(data_name):
    PATH = './DATA/{}.csv'.format(data_name)
    OUT_DF = './DATA/ml_{}.csv'.format(data_name)
    OUT_FEAT = './DATA/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './DATA/ml_{}_node.npy'.format(data_name)
    
    df, feat = preprocess(PATH)
    new_df = reindex(df)
    
    print(feat.shape)
    # Adds a zero-vector feature for index 0 (the padding index)
    # ensures feature matrix aligns with node indices starting at 1
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])
    # new shape : (num_edges + 1, feat_dim)
    
    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    # random features for nodes if needed
    # shape [num_nodes, d_feat]
    
    print(feat.shape)
    new_df.to_csv(OUT_DF) # all edges, reindexed (u,i,ts,label,idx)
    np.save(OUT_FEAT, feat) # edge features (+ zero row)
    np.save(OUT_NODE_FEAT, rand_feat) # node feature matrix (zeros)
    
def convert_tgat_to_tspear(dataset_name):
    tgat_csv      = f'tgat_data/original/ml_{dataset_name}.csv' # edge list
    edge_feat_npy = f'tgat_data/original/ml_{dataset_name}.npy' # edge features
    node_feat_npy = f'tgat_data/original/ml_{dataset_name}_node.npy' # node features

    df      = pd.read_csv(tgat_csv)   # columns: u,i,ts,label,idx
    edge_f  = np.load(edge_feat_npy)     # shape = [edges+1, feat_dim]
    node_f  = np.load(node_feat_npy)     # shape = [nodes+1, feat_dim]

    out_dir = f'DATA/{dataset_name}'
    os.makedirs(out_dir, exist_ok=True)

    edges_out = df[['u','i','ts']].copy()
    edges_out.columns = ['src','dst','time']# rename columns
    val_time, test_time = edges_out.time.quantile([0.70,0.85]).values
    # First 70 % -> training
    # Next 15 % -> validation
    # Last 15 % -> test

    # add rolling columns for TSPEAR
    edges_out['int_roll'] = 0        # internal CV fold 0 = none
    edges_out['ext_roll'] = -1 # external split (0=train, 1=val, 2=test)
    edges_out['ext_roll'] = edges_out['time'].apply(
        lambda t: 0 if   t <= val_time
                  else 1 if t <= test_time
                  else 2
    )
    edges_out['adv']      = 0 # adversarial flag (0 = clean edge)

    edges_out.to_csv(os.path.join(out_dir,'edges.csv'),
                     index=False) # now in Good format

    all_nodes = pd.unique(pd.concat([df.u, df.i]))
    N = max(all_nodes) + 1 # total number of unique nodes + 1
    # (because indices start at 1 for padding)
    raw_adj = [[] for _ in range(N)] # adjacency list
    # each element will hold (neighbor, time, edge_id)
    for _, row in df.iterrows():
        # For each edge (u,v), add v -> u and u -> v entries.
        u, v, t, idx = int(row.u), int(row.i), float(row.ts), int(row.idx)
        raw_adj[u].append((v, t, idx))
        raw_adj[v].append((u, t, idx))
        # Example:    
        #     Edge list:
        # u  i  ts   idx
        # 1  2  1.0   0
        # 1  3  2.0   1

        # raw_adj = [
        #   [],             # index 0 is padding
        #   [(2,1.0,0), (3,2.0,1)],  # neighbors of node 1
        #   [(1,1.0,0)],             # neighbors of node 2
        #   [(1,2.0,1)]              # neighbors of node 3
        # ]


    # Separate into lists
    ext_full_indices = [[] for _ in range(N)] # neighbor node IDs
    ext_full_ts      = [[] for _ in range(N)]
    ext_full_eid     = [[] for _ in range(N)] # edge IDs
    for u in range(N):
        for v, t, eid in raw_adj[u]:
            ext_full_indices[u].append(v)
            ext_full_ts     [u].append(t)
            ext_full_eid    [u].append(eid)
    # ext_full looks like:
    # | Node | Neighbor IDs | Time       | Edge IDs |
    # | 1    | [2, 3]       | [1.0, 2.0] | [0, 1]   |
    # | 2    | [1]          | [1.0]      | [0]      |
    # | 3    | [1]          | [2.0]      | [1]      |
    # ext_full_indices[1] = [2, 3], for node 1 (index 1)
    # ext_full_ts[1]      = [1.0, 2.0], for node 1
    # ext_full_eid[1]     = [0, 1], for node 1

    # build CSR arrays
    # instead of a list of lists, we compress everything 
    # into flat arrays with pointers that tell us where each 
    # node's neighbor list starts and ends

    indptr = np.zeros(N+1, dtype=np.int64)
    # needs one extra slot so each node i uses 
    # indptr[i]:indptr[i+1] to index its neighbors
    for i in range(N):
        indptr[i+1] = indptr[i] + len(ext_full_indices[i])
    # example inputs
    # ext_full_indices = [[], [2,3], [1], [1]], N = 4(nodes 0..3)
    # indptr = np.zeros(4+1, dtype=np.int64) , [0,0,0,0,0]
    # loop effect:
    # i=0 -> indptr[1] = 0 + len([]) = 0  -> [0,0,0,0,0]
    # i=1 -> indptr[2] = 0 + len([2,3]) = 2 -> [0,0,2,0,0]
    # i=2 -> indptr[3] = 2 + len([1]) = 3  -> [0,0,2,3,0]
    # i=3 -> indptr[4] = 3 + len([1]) = 4  -> [0,0,2,3,4]
    indices = np.concatenate(ext_full_indices).astype(np.int64)# [num_edges*2]
    ts      = np.concatenate(ext_full_ts).astype(np.float32) # [num_edges*2]
    eid     = np.concatenate(ext_full_eid).astype(np.int64) # [num_edges*2]

    # for example
    # | Node | Neighbors | Count | Cumulative offset |
    # | 0    | —         | 0     | 0                 |
    # | 1    | [2,3]     | 2     | 2                 |
    # | 2    | [1]       | 1     | 3                 |
    # | 3    | [1]       | 1     | 4                 |
    # so
    # indptr = [0, 0, 2, 3, 4]
    # indices = [2,3,1,1]
    # ts      = [1.0,2.0,1.0,2.0]
    # eid     = [0,1,0,1]
    # Now we can instantly slice
    # neighbors_of_1 = indices[indptr[1]:indptr[2]]  → [2,3]
    # times_of_1     = ts[indptr[1]:indptr[2]]       → [1.0,2.0]
    # edges_of_1     = eid[indptr[1]:indptr[2]]      → [0,1]
    # for 0 we get empty slices as expected since 
    # indices[indptr[0]:indptr[1]] = indices[0:0] → empty

    # assemble final graph dict
    g = {
        'ext_full_indices': ext_full_indices,
        'ext_full_ts':      ext_full_ts,
        'ext_full_eid':     ext_full_eid,
        'indptr':           indptr,
        'indices':          indices,
        'ts':               ts,
        'eid':              eid
    }

    with open(os.path.join(out_dir,'ext_full.pkl'),'wb') as f:
        pickle.dump(g, f) # save graph structure

    ef = torch.from_numpy(edge_f).float()
    torch.save(ef, os.path.join(out_dir,'edge_features.pt'))
    
    nf = torch.from_numpy(node_f).float()
    torch.save(nf, os.path.join(out_dir,'node_features.pt'))

    print("Converted TGAT files to TSPEAR format.")