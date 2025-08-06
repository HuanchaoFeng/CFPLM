import numpy as np
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_data(protein_path, rna_path):
    protein_vec = np.load(protein_path)
    rna_vec = np.load(rna_path)
    return protein_vec, rna_vec

def sum_cosine_similarity( protein_vec, rna_vec):
    protein_similarity_sum = cosine_similarity(protein_vec)
    rna_similarity_sum = cosine_similarity(rna_vec)
    return protein_similarity_sum, rna_similarity_sum

def top_k(matrix, k=20):
    # Quote from ccgnn's similarity structure
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)

def gather(protein_similarity_sum, rna_similarity_sum):
    protein_similarity_graph = top_k(protein_similarity_sum)
    rna_similarity_graph = top_k(rna_similarity_sum)

    protein_similarity_edge = np.stack(np.where(protein_similarity_graph > 0), axis=1)
    protein_similarity_val = protein_similarity_graph[protein_similarity_edge[:, 0], protein_similarity_edge[:, 1]]

    rna_similarity_edge = np.stack(np.where(rna_similarity_graph > 0), axis=1)
    rna_similarity_val = rna_similarity_graph[rna_similarity_edge[:, 0], rna_similarity_edge[:, 1]]                                                                                                                                                         
    return protein_similarity_edge, protein_similarity_val, rna_similarity_edge, rna_similarity_val

protein_vec, rna_vec = load_data('protein_vec.npy','RNA_vec.npy')
protein_similarity_sum, rna_similarity_sum = sum_cosine_similarity(protein_vec, rna_vec)
protein_similarity_edge, protein_similarity_val, rna_similarity_edge, rna_similarity_val = gather(protein_similarity_sum, rna_similarity_sum)
np.savez('protein_similarity.npz', edge=protein_similarity_edge, weight=protein_similarity_val)
np.savez('rna_similarity.npz', edge=rna_similarity_edge, weight=rna_similarity_val)

