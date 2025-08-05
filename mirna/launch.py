import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import to_undirected
from control import single_run  

def fold_run(device):
    seed = 2024  # 2024是我的模型最高的
    random.seed(seed)
    np.random.seed(seed)
    
    inter = np.load(f'/data/hcfeng/envs/get_rna/upload_model/mirna/interaction.npy')
    rna_vector = np.load(f'/data/hcfeng/envs/get_rna/upload_model/mirna/rna_vector.npy')
    protein_vector = np.load('/data/hcfeng/envs/get_rna/upload_model/mirna/protein_vector.npy')
    rna_similarity = np.load('/data/hcfeng/envs/get_rna/upload_model/mirna/rna_similarity.npz')
    rna_index = rna_similarity['edge']
    rna_val = rna_similarity['weight']
    protein_similarity = np.load(f'/data/hcfeng/envs/get_rna/upload_model/mirna/protein_similarity.npz')
    protein_index = protein_similarity['edge']
    protein_val = protein_similarity['weight']

    rna_vector = torch.tensor(rna_vector, dtype=torch.float).to(device)
    protein_vector = torch.tensor(protein_vector, dtype=torch.float).to(device)
    rna_index = torch.tensor(rna_index.T, dtype=torch.long).to(device)
    rna_val = torch.tensor(rna_val, dtype=torch.float).to(device)
    protein_index = torch.tensor(protein_index.T, dtype=torch.long).to(device)
    protein_val = torch.tensor(protein_val, dtype=torch.float).to(device)


    pos = inter[inter[:, 2] == 1][:, :2]
    neg = inter[inter[:, 2] == 0][:, :2]
    all_pairs = np.concatenate([pos, neg])
    all_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])


    indices = np.random.permutation(len(all_labels))
    all_pairs, all_labels = all_pairs[indices], all_labels[indices]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    totalROC, totalPR,toalAcc, totalPre, toalF1,toalREC = 0, 0,0,0,0,0

    for fold, (train_index, test_index) in enumerate(skf.split(all_pairs, all_labels), 1):
            train_x, test_x = all_pairs[train_index], all_pairs[test_index]
            train_y, test_y = all_labels[train_index], all_labels[test_index]

            train_x = torch.tensor(train_x, dtype=torch.float).to(device)
            train_y = torch.tensor(train_y, dtype=torch.float).to(device)
            test_x = torch.tensor(test_x, dtype=torch.float).to(device)
            test_y = torch.tensor(test_y, dtype=torch.float).to(device)

            train_pos_mask = (train_y == 1)
            train_pos_edges = train_x[train_pos_mask]
            inter_train = to_undirected(torch.tensor(train_pos_edges.T, dtype=torch.long).to(device))


            data_dict = {
                'inter_train': inter_train,
                'rna_vector': rna_vector, 'protein_vector': protein_vector,
                'rna_index': rna_index, 'rna_val': rna_val,
                'protein_index': protein_index, 'protein_val': protein_val,
                'train_x': train_x, 'train_y': train_y,
                'test_x': test_x, 'test_y': test_y
            }
    
            AUC, AUPR,ACC, PRE, F1,REC = single_run(data_dict,device)
            totalROC += AUC
            totalPR += AUPR
            toalAcc += ACC
            toalF1 += F1
            totalPre += PRE
            toalREC += REC

    print("average:"
                    f"ROC AUC={totalROC/5:.4f}, "
                    f"PRC AUC={totalPR/5:.4f}, "
                    f"Accuracy={toalAcc/5:.4f}, "
                    f"Precision={totalPre/5:.4f}, "
                    f"F1 Score={toalF1/5:.4f}, "
                    f"Recall={toalREC/5:.4f}")


fold_run('cuda:2')