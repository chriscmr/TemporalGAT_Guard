import os
import copy
import time
import itertools
import math

from tqdm import tqdm
import torch
import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KernelDensity

from .utils import (
    parse_config,
    to_dgl_blocks,
    node_to_dgl_blocks,
    prepare_input
)
from .utils_train import (
    create_model_mailbox_sampler,
    train_model_link_pred,
)
from .sampler import ParallelSampler


class TemporalAttack:
    def __init__(self, args):
        self.args = args
    
    def attack(self, orig_node_feats, orig_edge_feats, orig_g, orig_df, ptb_rate, args, seed=0, **kwargs):
        tot_num_ptb = int(len(orig_df) * ptb_rate)
        print(f'>> [{args.attack} attack] len_df: {len(orig_df)}, ptb_rate: {ptb_rate}, tot_num_ptb: {tot_num_ptb}')
        if ptb_rate <= 0.0:
            print(f'>> [{args.attack} attack] elapsed: 0s')
            return orig_node_feats, orig_edge_feats, orig_g, orig_df

        node_feats = None if orig_node_feats is None else orig_node_feats.clone().detach()
        edge_feats = None if orig_edge_feats is None else orig_edge_feats.clone().detach()
        g = copy.deepcopy(orig_g)
        df = orig_df.copy()

        if 'Unnamed: 0' not in df.columns:
            df['Unnamed: 0'] = np.arange(1, len(df) + 1, dtype=np.int64)
        is_hetero_graph = ('rel_type' in df.columns)

        hetero_flag = getattr(args, "hetero_attack", False)
        if isinstance(hetero_flag, str):
            hetero_flag = hetero_flag.lower() == "true"

        use_hetero_attack = bool(hetero_flag and is_hetero_graph and args.attack == "proposed")
        
        node_type = None
        type_to_nodes = None
        rel_to_src_types = None
        rel_to_dst_types = None

        if use_hetero_attack:
            base_dir = f"DATA/{args.data}"
            node_type_path = f"{base_dir}/node_type.npy"
            type_to_nodes_path = f"{base_dir}/type_to_nodes.npz"
            node_type = np.load(node_type_path)
            type_to_nodes_npz = np.load(type_to_nodes_path, allow_pickle=True)
            type_to_nodes = {int(k): type_to_nodes_npz[k].astype(np.int32) for k in type_to_nodes_npz.files}

            rel_to_src_types = (
                df.groupby("rel_type")["src"]
                .apply(lambda srcs: np.unique(node_type[srcs.values]))
                .to_dict()
            )
            rel_to_dst_types = (
                df.groupby("rel_type")["dst"]
                .apply(lambda dsts: np.unique(node_type[dsts.values]))
                .to_dict()
            )
        edge_rel_type = None
        if is_hetero_graph:
            rel_per_event = df['rel_type'].to_numpy(np.int32)
            edge_rel_type = rel_per_event[g['eid'] - 1]
        
        df['adv'] = np.zeros(len(df)).astype(np.int32)


        t_attack_start = time.time()
        train_df = df[df['ext_roll'] == 0]
        valid_df = df[df['ext_roll'] == 1]
        test_df = df[df['ext_roll'] == 2]

        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')

        if args.attack == "proposed":
            ####################### Load surrogate model #######################
            sample_param, memory_param, gnn_param, train_param = parse_config(args.surrogate)
            model, mailbox, sampler = create_model_mailbox_sampler(
                node_feats, edge_feats, g, df,
                sample_param, memory_param, gnn_param, train_param,
                is_hetero=is_hetero_graph
            )

            seed = seed if seed is not None else 0
            self.args.model_path = f'./MODEL/AAAI/NON_ROBUST/{args.data}/none/{args.surrogate}_seed_{seed}_best_from51.pt'
            if not os.path.isfile(self.args.model_path):
                raise NotImplementedError
            model.load_state_dict(torch.load(self.args.model_path))
            model.eval()

            if sampler is not None:
                sampler.reset()
            if mailbox is not None:
                mailbox.reset()
                model.memory_updater.last_updated_nid = None

        elif args.attack != "random":
            ####################### Make adjacency matrix ######################
            src_set, dst_set = set(df.src), set(df.dst)
            if args.data == "UCI" and args.data == "BITCOIN":
                src_set = src_set.union(dst_set)
                dst_set = dst_set.union(src_set)
            i = 0
            src_to_idx, idx_to_src = {}, {}
            dst_to_idx, idx_to_dst = {}, {}
            for s in src_set:
                src_to_idx[s] = i
                idx_to_src[i] = s
                i += 1
            for d in dst_set:
                if d in src_to_idx:
                    dst_to_idx[d] = src_to_idx[d]
                    idx_to_dst[src_to_idx[d]] = d
                else:
                    dst_to_idx[d] = i
                    idx_to_dst[i] = d
                    i += 1
            adj = np.zeros((i, i)).astype(np.int32) 
            graph = nx.Graph(adj)

        tot_num_ptb = 0

        for i_df, x_df in enumerate([train_df, valid_df, test_df]):
            seen_src_set = set()
            seen_dst_set = set()
            
            ptb_ts, ptb_src, ptb_dst = np.array([], dtype=np.float32), np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            ptb_rel = np.array([], dtype=np.int32) if use_hetero_attack else None
            ptb_edge_feats = np.array([]).reshape(-1, edge_feats.shape[1]) if edge_feats is not None else None

            for i_rows, rows in x_df.groupby(x_df.index // args.batch_size):
                if i_rows > 0:
                    num_ptb_batch = int(len(rows) * ptb_rate)
                    if num_ptb_batch <= 0:
                        continue
                    kde.fit(rows.time.values.reshape(-1, 1))
                    ptb_ts_batch = np.trunc(kde.sample(num_ptb_batch).reshape(-1)).astype(np.float32)
                    ptb_ts_batch = np.clip(ptb_ts_batch, rows.time.iloc[0], rows.time.iloc[-1])

                    t_batch_start = time.time()
                    ptb_rel_batch = None
                    ptb_edge_feats_batch = None
                    if edge_feats is not None:
                        kde.fit(edge_feats[prev_rows['Unnamed: 0'].values])
                        ptb_edge_feats_batch = np.round(kde.sample(num_ptb_batch), 4)

                    if args.attack == "random":
                        ptb_src_batch = np.random.choice(list(row_src_set), num_ptb_batch, replace=True)
                        ptb_dst_batch = np.random.choice(list(row_dst_set), num_ptb_batch, replace=True)
                        if args.data == "UCI" or args.data == "BITCOIN":
                            mask = ptb_src_batch == ptb_dst_batch
                            while mask.sum() != 0:
                                ptb_src_batch[mask] = np.random.choice(list(row_src_set), mask.sum(), replace=True)
                                ptb_dst_batch[mask] = np.random.choice(list(row_dst_set), mask.sum(), replace=True)
                                mask = ptb_src_batch == ptb_dst_batch
                
                    elif args.attack == "preference" or args.attack == "jaccard" or args.attack == "degree" or args.attack == "pagerank":
                        if args.attack == "preference":
                            src_dst_pair = np.array(list(itertools.product(list(row_src_set), list(row_dst_set)))).astype(np.int32)
                            score = list(nx.preferential_attachment(graph, src_dst_pair))
                            src_dst_score = np.array([i for _, _, i in score])
                            # src_dst_score = np.log(src_dst_score + 1)
                        elif args.attack == "jaccard": # Not for bipartite graph
                            src_dst_pair = np.array(list(itertools.product(list(row_src_set), list(row_dst_set)))).astype(np.int32)
                            score = list(nx.jaccard_coefficient(graph, src_dst_pair))
                            src_dst_score = np.array([i for _, _, i in score])
                        elif args.attack == "degree":
                            deg = adj.sum(1)
                            src_dst_pair = np.array(list(itertools.product(list(row_src_set), list(row_dst_set)))).astype(np.int32)
                            src_dst_score = deg[src_dst_pair[:, 0]] + deg[src_dst_pair[:, 1]]
                        elif args.attack == "pagerank":
                            pr = nx.pagerank(graph, max_iter=10, tol=1e-3)
                            pr = np.array([pr[i] for i in range(graph.number_of_nodes())]).astype(np.float32)
                            src_dst_pair = np.array(list(itertools.product(list(row_src_set), list(row_dst_set)))).astype(np.int32)
                            src_dst_score = pr[src_dst_pair[:, 0]] + pr[src_dst_pair[:, 1]]

                        num_hungarian = 0
                        cost = src_dst_score.reshape(len(row_src_set), len(row_dst_set))
                        for elem in inter: ## Remove self-interactions from candidates
                            cost[np.where(row_src_array == elem)[0], np.where(row_dst_array == elem)[0]] = 999999

                        cost1 = cost.copy()
                        row_cand, col_cand = np.array([]).astype(np.int32), np.array([]).astype(np.int32)
                        while len(row_cand) < num_ptb_batch:
                            row_ind, col_ind = linear_sum_assignment(cost)
                            row_cand = np.concatenate((row_cand, row_ind)) 
                            col_cand = np.concatenate((col_cand, col_ind))
                            cost[row_ind, col_ind] = 999999
                            num_hungarian += 1

                        for i_h in range(args.xpool):
                            row_ind, col_ind = linear_sum_assignment(cost)
                            row_cand = np.concatenate((row_cand, row_ind)) 
                            col_cand = np.concatenate((col_cand, col_ind))
                            cost[row_ind, col_ind] = 999999
                            num_hungarian += 1

                        idx_batch = cost1[row_cand, col_cand].argsort()[:num_ptb_batch]
                        ptb_src_batch = np.array([idx_to_src[i] for i in row_src_array[row_cand[idx_batch]]]).astype(np.int32)
                        ptb_dst_batch = np.array([idx_to_dst[j] for j in row_dst_array[col_cand[idx_batch]]]).astype(np.int32)
                        assert num_ptb_batch == len(ptb_src_batch) and num_ptb_batch == len(ptb_dst_batch)
                
                    elif args.attack == "proposed":
                        ######################## Edge scores #######################
                        if not use_hetero_attack:
                            src_dst_pair = np.array(list(itertools.product(row_src_array, row_dst_array))).astype(np.int32)
                            root_nodes = src_dst_pair.T.reshape(1, -1).squeeze()
                            ts = ptb_ts_batch.min() * np.ones_like(root_nodes).astype(np.float32)

                            if sampler is not None:
                                sampler.sample(root_nodes, ts)
                                ret = sampler.get_ret()

                            if gnn_param['arch'] != 'identity':
                                mfgs = to_dgl_blocks(ret, sample_param['history'])
                            else:
                                mfgs = node_to_dgl_blocks(root_nodes, ts)

                            mfgs = prepare_input(mfgs, node_feats, edge_feats, edge_rel_type)
                            if mailbox is not None:
                                mailbox.prep_input_mails(mfgs[0])

                            with torch.no_grad():
                                src_dst_score, _ = model(mfgs, neg_samples=0)

                            num_hungarian = 0
                            if not args.use_hungarian:
                                idx_batch = src_dst_score.squeeze().topk(num_ptb_batch, largest=False).indices.cpu().numpy()
                                ptb_src_batch = src_dst_pair[idx_batch][:, 0]
                                ptb_dst_batch = src_dst_pair[idx_batch][:, 1]
                            else:
                                cost = src_dst_score.reshape(len(row_src_set), len(row_dst_set)).cpu().numpy()
                                for elem in inter:
                                    cost[np.where(row_src_array == elem)[0], np.where(row_dst_array == elem)[0]] = 999999

                                cost1 = cost.copy()
                                row_cand, col_cand = np.array([]).astype(np.int32), np.array([]).astype(np.int32)
                                while len(row_cand) < num_ptb_batch:
                                    row_ind, col_ind = linear_sum_assignment(cost)
                                    row_cand = np.concatenate((row_cand, row_ind))
                                    col_cand = np.concatenate((col_cand, col_ind))
                                    cost[row_ind, col_ind] = 999999
                                    num_hungarian += 1

                                for i_h in range(args.xpool):
                                    row_ind, col_ind = linear_sum_assignment(cost)
                                    row_cand = np.concatenate((row_cand, row_ind))
                                    col_cand = np.concatenate((col_cand, col_ind))
                                    cost[row_ind, col_ind] = 999999
                                    num_hungarian += 1

                                idx_batch = cost1[row_cand, col_cand].argsort()[:num_ptb_batch]
                                ptb_src_batch = row_src_array[row_cand[idx_batch]]
                                ptb_dst_batch = row_dst_array[col_cand[idx_batch]]

                            assert num_ptb_batch == len(ptb_src_batch) and num_ptb_batch == len(ptb_dst_batch)

                        else:
                            # Extension: relation-by-relation
                            ptb_src_parts = []
                            ptb_dst_parts = []
                            ptb_ts_parts = []
                            ptb_rel_parts = []
                            ptb_edge_feat_parts = [] if edge_feats is not None else None
                            num_hungarian = 0

                            rel_groups = list(rows.groupby("rel_type"))
                            if len(rel_groups) == 0:
                                ptb_src_batch = np.array([], dtype=np.int32)
                                ptb_dst_batch = np.array([], dtype=np.int32)
                                ptb_ts_batch = np.array([], dtype=np.float32)
                                ptb_rel_batch = np.array([], dtype=np.int32)
                                if edge_feats is not None:
                                    ptb_edge_feats_batch = np.empty((0, edge_feats.shape[1]), dtype=np.float32)
                            else:
                                # allocate perturbation budget proportionally to relation frequency in this batch
                                rel_sizes = np.array([len(rel_rows) for _, rel_rows in rel_groups], dtype=np.float64)
                                rel_budget_float = rel_sizes / rel_sizes.sum() * num_ptb_batch
                                rel_budget = np.floor(rel_budget_float).astype(int)
                                remainder = num_ptb_batch - rel_budget.sum()
                                if remainder > 0:
                                    order = np.argsort(-(rel_budget_float - rel_budget))
                                    rel_budget[order[:remainder]] += 1

                                for (rel_val, rel_rows), num_ptb_rel in zip(rel_groups, rel_budget):
                                    if num_ptb_rel <= 0:
                                        continue

                                    # timestamps from this relation bucket
                                    kde.fit(rel_rows.time.values.reshape(-1, 1))
                                    ptb_ts_rel = np.trunc(kde.sample(num_ptb_rel).reshape(-1)).astype(np.float32)
                                    ptb_ts_rel = np.clip(ptb_ts_rel, rel_rows.time.iloc[0], rel_rows.time.iloc[-1]).astype(np.float32)

                                    # edge features from previous rows of same relation (fallback to prev_rows)
                                    if edge_feats is not None:
                                        prev_rel_rows = prev_rows[prev_rows['rel_type'] == rel_val]
                                        if len(prev_rel_rows) > 0:
                                            kde.fit(edge_feats[prev_rel_rows['Unnamed: 0'].values])
                                        else:
                                            kde.fit(edge_feats[prev_rows['Unnamed: 0'].values])
                                        ptb_edge_feats_rel = np.round(kde.sample(num_ptb_rel), 4).astype(np.float32)
                                    #  type-compatible candidate pools inside this relation bucket
                                    src_type_list = rel_to_src_types[int(rel_val)]
                                    dst_type_list = rel_to_dst_types[int(rel_val)]

                                    src_type_nodes = []
                                    for t in src_type_list:
                                        t = int(t)
                                        if t in type_to_nodes:
                                            src_type_nodes.append(type_to_nodes[t])

                                    dst_type_nodes = []
                                    for t in dst_type_list:
                                        t = int(t)
                                        if t in type_to_nodes:
                                            dst_type_nodes.append(type_to_nodes[t])

                                    if len(src_type_nodes) > 0:
                                        rel_src_pool = np.unique(np.concatenate(src_type_nodes)).astype(np.int32)
                                    else:
                                        rel_src_pool = np.unique(rel_rows.src.values).astype(np.int32)

                                    if len(dst_type_nodes) > 0:
                                        rel_dst_pool = np.unique(np.concatenate(dst_type_nodes)).astype(np.int32)
                                    else:
                                        rel_dst_pool = np.unique(rel_rows.dst.values).astype(np.int32)

                                    # keep attack causal: only use nodes already seen earlier in this split
                                    if len(seen_src_set) > 0:
                                        rel_src_array = np.array(
                                            sorted(set(rel_src_pool).intersection(seen_src_set)),
                                            dtype=np.int32
                                        )
                                    else:
                                        rel_src_array = np.unique(rel_rows.src.values).astype(np.int32)

                                    if len(seen_dst_set) > 0:
                                        rel_dst_array = np.array(
                                            sorted(set(rel_dst_pool).intersection(seen_dst_set)),
                                            dtype=np.int32
                                        )
                                    else:
                                        rel_dst_array = np.unique(rel_rows.dst.values).astype(np.int32)

                                    # fallback if the filtered pools are empty
                                    if len(rel_src_array) == 0:
                                        rel_src_array = np.unique(rel_rows.src.values).astype(np.int32)
                                    if len(rel_dst_array) == 0:
                                        rel_dst_array = np.unique(rel_rows.dst.values).astype(np.int32)

                                    # optional candidate cap for speed
                                    max_src_cands = getattr(args, "max_src_cands", 200)
                                    max_dst_cands = getattr(args, "max_dst_cands", 200)

                                    if len(rel_src_array) > max_src_cands:
                                        rel_src_array = np.random.choice(
                                            rel_src_array, size=max_src_cands, replace=False
                                        ).astype(np.int32)

                                    if len(rel_dst_array) > max_dst_cands:
                                        rel_dst_array = np.random.choice(
                                            rel_dst_array, size=max_dst_cands, replace=False
                                        ).astype(np.int32)

                                    rel_inter = set(rel_src_array).intersection(set(rel_dst_array))

                                    if len(rel_src_array) == 0 or len(rel_dst_array) == 0:
                                        continue

                                    src_dst_pair = np.array(list(itertools.product(rel_src_array, rel_dst_array))).astype(np.int32)
                                    root_nodes = src_dst_pair.T.reshape(1, -1).squeeze()
                                    ts = ptb_ts_rel.min() * np.ones_like(root_nodes).astype(np.float32)

                                    if sampler is not None:
                                        sampler.sample(root_nodes, ts)
                                        ret = sampler.get_ret()

                                    if gnn_param['arch'] != 'identity':
                                        mfgs = to_dgl_blocks(ret, sample_param['history'])
                                    else:
                                        mfgs = node_to_dgl_blocks(root_nodes, ts)

                                    mfgs = prepare_input(mfgs, node_feats, edge_feats, edge_rel_type)
                                    if mailbox is not None:
                                        mailbox.prep_input_mails(mfgs[0])

                                    with torch.no_grad():
                                        src_dst_score, _ = model(mfgs, neg_samples=0)

                                    if not args.use_hungarian:
                                        idx_rel = src_dst_score.squeeze().topk(num_ptb_rel, largest=False).indices.cpu().numpy()
                                        ptb_src_rel = src_dst_pair[idx_rel][:, 0]
                                        ptb_dst_rel = src_dst_pair[idx_rel][:, 1]
                                    else:
                                        cost = src_dst_score.reshape(len(rel_src_array), len(rel_dst_array)).cpu().numpy()
                                        for elem in rel_inter:
                                            cost[np.where(rel_src_array == elem)[0], np.where(rel_dst_array == elem)[0]] = 999999

                                        cost1 = cost.copy()
                                        row_cand, col_cand = np.array([]).astype(np.int32), np.array([]).astype(np.int32)

                                        while len(row_cand) < num_ptb_rel:
                                            row_ind, col_ind = linear_sum_assignment(cost)
                                            row_cand = np.concatenate((row_cand, row_ind))
                                            col_cand = np.concatenate((col_cand, col_ind))
                                            cost[row_ind, col_ind] = 999999
                                            num_hungarian += 1

                                        for i_h in range(args.xpool):
                                            row_ind, col_ind = linear_sum_assignment(cost)
                                            row_cand = np.concatenate((row_cand, row_ind))
                                            col_cand = np.concatenate((col_cand, col_ind))
                                            cost[row_ind, col_ind] = 999999
                                            num_hungarian += 1

                                        idx_rel = cost1[row_cand, col_cand].argsort()[:num_ptb_rel]
                                        ptb_src_rel = rel_src_array[row_cand[idx_rel]]
                                        ptb_dst_rel = rel_dst_array[col_cand[idx_rel]]

                                    ptb_rel_rel = np.full(len(ptb_src_rel), rel_val, dtype=np.int32)

                                    ptb_src_parts.append(ptb_src_rel)
                                    ptb_dst_parts.append(ptb_dst_rel)
                                    ptb_ts_parts.append(ptb_ts_rel[:len(ptb_src_rel)])
                                    ptb_rel_parts.append(ptb_rel_rel)

                                    if edge_feats is not None:
                                        ptb_edge_feat_parts.append(ptb_edge_feats_rel[:len(ptb_src_rel)])

                                ptb_src_batch = np.concatenate(ptb_src_parts) if len(ptb_src_parts) > 0 else np.array([], dtype=np.int32)
                                ptb_dst_batch = np.concatenate(ptb_dst_parts) if len(ptb_dst_parts) > 0 else np.array([], dtype=np.int32)
                                ptb_ts_batch = np.concatenate(ptb_ts_parts) if len(ptb_ts_parts) > 0 else np.array([], dtype=np.float32)
                                ptb_rel_batch = np.concatenate(ptb_rel_parts) if len(ptb_rel_parts) > 0 else np.array([], dtype=np.int32)

                                if edge_feats is not None:
                                    ptb_edge_feats_batch = (
                                        np.concatenate(ptb_edge_feat_parts)
                                        if len(ptb_edge_feat_parts) > 0
                                        else np.empty((0, edge_feats.shape[1]), dtype=np.float32)
                                    )
                    
                    ptb_ts = np.concatenate((ptb_ts, ptb_ts_batch))
                    ptb_src = np.concatenate((ptb_src, ptb_src_batch))
                    ptb_dst = np.concatenate((ptb_dst, ptb_dst_batch))
                    if use_hetero_attack and ptb_rel_batch is not None:
                        ptb_rel = np.concatenate((ptb_rel, ptb_rel_batch))
                    if edge_feats is not None:
                        ptb_edge_feats = np.concatenate((ptb_edge_feats, ptb_edge_feats_batch))
                    num_src_id = len(set(ptb_src_batch)) 
                    num_dst_id = len(set(ptb_dst_batch))
                    t_batch_elapsed = time.time() - t_batch_start
                    if args.attack == "random":
                        print(f'* [Batch {i_df}-{i_rows}] t_batch_elapsed: {t_batch_elapsed:.4f}s, num_ptb_batch: {num_ptb_batch}, num_src_id: {num_src_id}/{len(row_src_set)}, num_dst_id: {num_dst_id}/{len(row_dst_set)}')
                    elif args.attack == "proposed":
                        print(f'* [Batch {i_df}-{i_rows}] t_batch_elapsed: {t_batch_elapsed:.4f}s, num_ptb_batch: {num_ptb_batch}, xpool: {args.xpool}, num_hungarian: {num_hungarian}, num_src_id: {num_src_id}/{len(row_src_set)}, num_dst_id: {num_dst_id}/{len(row_dst_set)}')
                    else:
                        print(f'* [Batch {i_df}-{i_rows}] t_batch_elapsed: {t_batch_elapsed:.4f}s, num_ptb_batch: {num_ptb_batch}, num_hungarian: {num_hungarian}, num_src_id: {num_src_id}/{len(row_src_set)}, num_dst_id: {num_dst_id}/{len(row_dst_set)}')

                prev_rows = rows
                seen_src_set.update(rows.src.values.tolist())
                seen_dst_set.update(rows.dst.values.tolist())
                if args.attack == "random":
                    row_src_set = set(rows.src)
                    row_dst_set = set(rows.dst)
                    if args.data == "UCI" or args.data == "BITCOIN":
                        row_src_set = row_src_set.union(row_dst_set)
                        row_dst_set = row_dst_set.union(row_src_set)
                elif args.attack == "preference" or args.attack == "jaccard" or args.attack == "degree" or args.attack == "pagerank":
                    src_idx = [src_to_idx[s] for s in rows.src.values]
                    dst_idx = [dst_to_idx[d] for d in rows.dst.values]
                    for s, d in zip(src_idx, dst_idx):
                        adj[s, d] = 1
                        adj[d, s] = 1
                    graph.add_edges_from(list(zip(src_idx, dst_idx)))
                    row_src_set = set(src_idx)
                    row_dst_set = set(dst_idx)
                    if args.data == "UCI" or args.data == "BITCOIN":
                        row_src_set = row_src_set.union(row_dst_set)
                        row_dst_set = row_dst_set.union(row_src_set)
                    row_src_array = np.array(list(row_src_set)).astype(np.int32)
                    row_dst_array = np.array(list(row_dst_set)).astype(np.int32)
                    inter = row_src_set.intersection(row_dst_set)
                elif args.attack == "proposed":
                    root_nodes = np.concatenate([rows.src.values, rows.dst.values]).astype(np.int32)
                    ts = np.concatenate([rows.time.values, rows.time.values]).astype(np.float32)
                    if sampler is not None:
                        sampler.sample(root_nodes, ts)
                        ret = sampler.get_ret()
                    if gnn_param['arch'] != 'identity':
                        mfgs = to_dgl_blocks(ret, sample_param['history'])
                    else:
                        mfgs = node_to_dgl_blocks(root_nodes, ts)
                    mfgs = prepare_input(mfgs, node_feats, edge_feats, edge_rel_type)
                    if mailbox is not None:
                        mailbox.prep_input_mails(mfgs[0])
                    with torch.no_grad():
                        _, _ = model(mfgs, neg_samples=0)

                    if mailbox is not None:
                        eid = rows['Unnamed: 0'].values
                        mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                        block = None
                        if memory_param['deliver_to'] == 'neighbors':
                            block = to_dgl_blocks(ret, sample_param['history'], reverse=True)[0][0]
                        rel_emb = None
                        if is_hetero_graph:
                            rel_ids = rows['rel_type'].to_numpy().astype(np.int64)   # one rel id per true edge
                            rel_ids = torch.from_numpy(rel_ids).to(model.mail_rel_emb.weight.device)
                            rel_emb = model.mail_rel_emb(rel_ids)                    # shape [E, dim_mail_rel]

                        mailbox.update_mailbox(
                            model.memory_updater.last_updated_nid,
                            model.memory_updater.last_updated_memory,
                            root_nodes, ts, mem_edge_feats, block,
                            rel_emb=rel_emb,
                            neg_samples=0
                        )
                        mailbox.update_memory(
                            model.memory_updater.last_updated_nid,
                            model.memory_updater.last_updated_memory,
                            root_nodes, model.memory_updater.last_updated_ts,
                            neg_samples=0
                        )

                    row_src_set = set(rows.src)
                    row_dst_set = set(rows.dst)
                    if args.data == "UCI" or args.data == "BITCOIN":
                        row_src_set = row_src_set.union(row_dst_set)
                        row_dst_set = row_dst_set.union(row_src_set)
                    row_src_array = np.array(list(row_src_set)).astype(np.int32)
                    row_dst_array = np.array(list(row_dst_set)).astype(np.int32)
                    inter = row_src_set.intersection(row_dst_set)


            num_ptb = len(ptb_ts)
            tot_num_ptb += num_ptb 
            ptb_dict = {
                'Unnamed: 0': np.arange(len(df) + 1, len(df) + num_ptb + 1), 
                'src': ptb_src, 
                'dst': ptb_dst, 
                'time': ptb_ts, 
                'int_roll': 0, 
                'ext_roll': i_df,
                'adv': 1
            }
            if use_hetero_attack:
                ptb_dict['rel_type'] = ptb_rel
            ptb_df = pd.DataFrame(ptb_dict)
            ptb_edge_feats = torch.from_numpy(ptb_edge_feats).to(dtype=torch.float32) if ptb_edge_feats is not None else None
            node_feats, edge_feats, g, df = self.add_perturbations(node_feats, edge_feats, g, df, ptb_edge_feats, ptb_df)
            if is_hetero_graph:
                rel_per_event = df['rel_type'].to_numpy(np.int32)
                edge_rel_type = rel_per_event[g['eid'] - 1]
        
        t_attack_elapsed = time.time() - t_attack_start
        print(f'>> [{args.attack} attack] ptb_rate: {ptb_rate}, tot_num_ptb: {tot_num_ptb}, elapsed: {t_attack_elapsed:.4f}s')
        return node_feats, edge_feats, g, df

    def add_perturbations(self, node_feats, edge_feats, g, df, ptb_edge_feats, ptb_df, verbose=False):
        df = pd.concat([df, ptb_df], ignore_index=True)
        df = df.sort_values('time')
        edge_feats = torch.vstack((edge_feats, ptb_edge_feats)) if ptb_edge_feats is not None else edge_feats
        df = df.reset_index(drop=True)

        num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
        ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
        ext_full_indices = g['ext_full_indices']
        ext_full_ts = g['ext_full_ts']
        ext_full_eid = g['ext_full_eid']
        ext_full_rel_type = g['ext_full_rel_type'] if 'ext_full_rel_type' in g else None

        for i, row in tqdm(ptb_df.iterrows(), total=len(ptb_df), disable=not verbose):
            src = int(row['src'])
            dst = int(row['dst'])
            idx = int(row['Unnamed: 0'])

            ext_full_indices[src].append(dst)
            ext_full_ts[src].append(row['time'])
            ext_full_eid[src].append(idx)

            ext_full_indices[dst].append(src)
            ext_full_ts[dst].append(row['time'])
            ext_full_eid[dst].append(idx)

            if ext_full_rel_type is not None and 'rel_type' in ptb_df.columns:
                r = int(row['rel_type'])
                ext_full_rel_type[src].append(r)
                ext_full_rel_type[dst].append(r)

        for i in tqdm(range(num_nodes), disable=True):
            ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])
        
        def ext_sort(i, indices, t, eid, rel = None):
            idx = np.argsort(t[i])
            indices[i] = np.array(indices[i])[idx].tolist()
            t[i] = np.array(t[i])[idx].tolist()
            eid[i] = np.array(eid[i])[idx].tolist()
            if rel is not None:
                rel[i] = np.array(rel[i])[idx].tolist()
        
        for i in tqdm(range(num_nodes), disable=True):
            ext_sort(i, ext_full_indices, ext_full_ts, ext_full_eid, ext_full_rel_type)

        np_ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
        np_ext_full_ts = np.array(list(itertools.chain(*ext_full_ts))) 
        np_ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))
        np_ext_full_rel_type = np.array(list(itertools.chain(*ext_full_rel_type))) if ext_full_rel_type is not None else None

        def tsort(i, indptr, indices, t, eid, rel=None):
            beg = indptr[i]
            end = indptr[i + 1]
            sidx = np.argsort(t[beg:end])
            indices[beg:end] = indices[beg:end][sidx]
            t[beg:end] = t[beg:end][sidx]
            eid[beg:end] = eid[beg:end][sidx]
            if rel is not None:
                rel[beg:end] = rel[beg:end][sidx]

        for i in tqdm(range(ext_full_indptr.shape[0] - 1), disable=True):
            tsort(i, ext_full_indptr, np_ext_full_indices, np_ext_full_ts, np_ext_full_eid, np_ext_full_rel_type)

        g['indptr'] = ext_full_indptr
        g['indices'] = np_ext_full_indices
        g['ts'] = np_ext_full_ts
        g['eid'] = np_ext_full_eid

        if np_ext_full_rel_type is not None:
            g['rel_type'] = np_ext_full_rel_type
            g['ext_full_rel_type'] = ext_full_rel_type
        return node_feats, edge_feats, g, df