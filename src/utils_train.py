from collections import deque
# import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from deeprobust.graph.defense.pgd import PGD, prox_operators

from .modules import GeneralModel, EstimateAdj
from .sampler import ParallelSampler, NegLinkSampler, NegLinkSamplerDST
from .utils import (
    to_dgl_blocks,
    node_to_dgl_blocks,
    prepare_input,
)
from pympler.tracker import SummaryTracker

def create_model_mailbox_sampler(node_feats, edge_feats, g, df, sample_param, gnn_param, train_param):
    gnn_dim_node = 0 if node_feats is None else node_feats.shape[1]
    gnn_dim_edge = 0 if edge_feats is None else edge_feats.shape[1]
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True

    model = GeneralModel(gnn_dim_node, gnn_dim_edge, sample_param, gnn_param, train_param, combined=combine_first).cuda()
    
    if 'all_on_gpu' in train_param and train_param['all_on_gpu']:
        if node_feats is not None:
            node_feats = node_feats.cuda()
        if edge_feats is not None:
            edge_feats = edge_feats.cuda()
    
    sampler = None
    if not ('no_sample' in sample_param and sample_param['no_sample']):
        sampler = ParallelSampler(g['indptr'], g['indices'], g['eid'], g['ts'].astype(np.float32),
                                  sample_param['num_thread'], 1, sample_param['layer'], sample_param['neighbor'],
                                  sample_param['strategy']=='recent', sample_param['prop_time'],
                                  sample_param['history'], float(sample_param['duration']))
    
    return model, sampler

def train_model_link_pred(node_feats, edge_feats, g, df, model, sampler, sample_param, gnn_param, train_param, args, seed=0):
    # node_feats node features tensor [N+1, d_n]
    # edge_feats tensor [E+1, d_e]
    # g dict the ext_full.pkl graph
    # df pd.DataFrame edges.csv (src, dst, time, ext_roll, adv)
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True
        # merge duplicate neighbors in the first layer

    if args.data != "UCI" and args.data != "BITCOIN":
        neg_link_sampler = NegLinkSamplerDST(df.dst.values)
    else:
        # UCI and Bitcoin have special ID distributions
        #  sample negatives from all nodes
        src_set = set(df.src.values)
        dst_set = set(df.dst.values)
        node_set = src_set.union(dst_set)
        neg_link_sampler = NegLinkSamplerDST(node_set)

    criterion = torch.nn.BCEWithLogitsLoss()
    # input raw logits and applies sigmoid internally
    optimizer = torch.optim.Adam(model.parameters(), lr=train_param['lr'])

    train_df = df[df['ext_roll'] == 0]  # training set  

    # best_val_auc = -1
    for e in range(1, train_param['epoch'] + 1):
        time_sample = 0
        time_prep = 0
        time_tot = 0
        total_loss = 0

        # training
        model.train()
        if sampler is not None:
            sampler.reset()
            # reset sampler memory at the beginning of each epoch

        for _, rows in train_df.groupby(train_df.index // train_param['batch_size']):
            # row is a mini-batch of edges
            # variable are rows.src, rows.dst, rows.time
            # each has shape [batch_size]
            t_tot_s = time.time()
            root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v4(rows.dst.values, neg_samples=1)]).astype(np.int32)
            # we want the embeddings of src, pos dst and neg dst
            ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
            # same timestamps for src, pos dst and neg dst
            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    # If negative sampling is disabled for the sampler, 
                    # only sample src & pos dst
                    pos_root_end = root_nodes.shape[0] * 2 // 3 # first two thirds are src and pos dst
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                    # Sample temporal neighbors for all nodes (src, dst, neg)
                ret = sampler.get_ret() # list of sampler results (one per layer × history)
                # ret = [r0, r1] if 2 layer and one history
                # note that in r1 we sampled neighbors for source node from r0
                time_sample += ret[0].sample_time()
            t_prep_s = time.time()
            if gnn_param['arch'] != 'identity':
                # Convert sampler results to DGL blocks
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            # mfgs is a list of lists of DGL blocks
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            
            time_prep += time.time() - t_prep_s

            ## Default Loss function
            optimizer.zero_grad()
            if args.robust == "none":
                pred_pos, pred_neg = model(mfgs)
                loss = criterion(pred_pos, torch.ones_like(pred_pos))
                loss += criterion(pred_neg, torch.zeros_like(pred_neg))
            total_loss += float(loss) * train_param['batch_size']
            loss.backward()
            optimizer.step()

            t_prep_s = time.time()            
            time_prep += time.time() - t_prep_s
            time_tot += time.time() - t_tot_s

        model_path = f'./MODEL/{args.exp_id}/{args.exp_file}_seed_{seed}_{e}.pt'
        torch.save(model.state_dict(), model_path)

        t_val_s = time.time()
        val_ap, val_auc, val_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, sampler, sample_param, gnn_param, train_param, args, negs=1, mode='val', evaluation=False)        
        time_val = time.time() - t_val_s
        # args.logger.debug(f'Epoch: {e}, train_loss: {total_loss:.1f}, total_time: {(time_tot + time_val):.2f}s (sample: {time_sample:.2f}s, prep: {time_prep:.2f}s, val: {time_val:.2f}s)')
        args.logger.debug(f'Epoch: {e}, train_loss: {total_loss:.1f}, val_ap: {val_ap:.4f}, val_auc: {val_auc:.4f}, total_time: {(time_tot + time_val):.2f}s (sample: {time_sample:.2f}s, prep: {time_prep:.2f}s, val: {time_val:.2f}s)')

    model.eval()
    # val_ap, val_auc, val_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, mailbox, sampler, sample_param, memory_param, gnn_param, train_param, args, negs=1, mode='val', evaluation=False)
    test_ap, test_auc, test_hit = link_pred_evaluation(node_feats, edge_feats, g, df, model, sampler, sample_param, gnn_param, train_param, args, negs=1, mode='test', evaluation=False)
    torch.cuda.empty_cache()
    
    return val_ap, val_auc, val_hit, test_ap, test_auc, test_hit

@torch.no_grad()
def link_pred_evaluation(node_feats, edge_feats, g, df, model, sampler, sample_param, gnn_param, train_param, args, negs=1, mode='val', seed=None, evaluation=False):
    combine_first = False
    if 'combine_neighs' in train_param and train_param['combine_neighs']:
        combine_first = True

    if args.data != "UCI" and args.data != "BITCOIN":
        neg_link_sampler = NegLinkSamplerDST(df.dst.values, seed=seed)
    else:
        src_set = set(df.src.values)
        dst_set = set(df.dst.values)
        node_set = src_set.union(dst_set)
        neg_link_sampler = NegLinkSamplerDST(node_set, seed=seed)

    neg_samples = 1
    model.eval()
    aps = list()
    aucs_mrrs = list()
    hits = list()

    if mode == 'val':
        eval_df = df[df['ext_roll'] == 1]
        neg_samples = negs
        if evaluation and args.data == "MOOC": 
            neg_samples = len(neg_link_sampler.dsts) - 1
    elif mode == 'test':
        eval_df = df[df['ext_roll'] == 2]
        neg_samples = negs
        if evaluation and args.data == "MOOC": 
            neg_samples = len(neg_link_sampler.dsts) - 1
    elif mode == 'train':
        eval_df = df[df['ext_roll'] == 0]

    with torch.no_grad():
        for _, rows in eval_df.groupby(eval_df.index // train_param['batch_size']):
            if evaluation and args.data == "MOOC": 
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v3(rows.dst.values)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            elif evaluation and args.data != "MOOC":
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v2(rows.dst.values, neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)
            else:
                root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample_v4(rows.dst.values, neg_samples)]).astype(np.int32)
                ts = np.tile(rows.time.values, neg_samples + 2).astype(np.float32)

            if sampler is not None:
                if 'no_neg' in sample_param and sample_param['no_neg']:
                    pos_root_end = len(rows) * 2
                    sampler.sample(root_nodes[:pos_root_end], ts[:pos_root_end])
                else:
                    sampler.sample(root_nodes, ts)
                ret = sampler.get_ret()
            if gnn_param['arch'] != 'identity':
                mfgs = to_dgl_blocks(ret, sample_param['history'])
            else:
                mfgs = node_to_dgl_blocks(root_nodes, ts)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=combine_first)
            
            with torch.no_grad():
                pred_pos, pred_neg = model(mfgs, neg_samples=neg_samples)
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).squeeze().sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)

            #### TODO: only test for non adv
            if args.attack != "none" and mode == "test":
                pos_mask = (rows['adv'] == 0).values
                neg_mask = np.tile(pos_mask, neg_samples)
                pos_neg_mask = np.concatenate((pos_mask, neg_mask))
                pred_pos = pred_pos[pos_mask]
                pred_neg = pred_neg[neg_mask]
                y_pred = y_pred[pos_neg_mask]
                y_true = y_true[pos_neg_mask]

            aps.append(average_precision_score(y_true, y_pred))
            if neg_samples > 1:
                ranks = torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1
                aucs_mrrs.append(torch.reciprocal(ranks).type(torch.float))
                hits.append(ranks <= 10)
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))
            
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
        hit10 = float(torch.cat(hits).sum() / len(torch.cat(hits)))
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
        hit10 = 0
    return ap, auc_mrr, hit10