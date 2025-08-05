from model import read_protein,ESM

ProSeq=read_protein(path)
proteinVec=ESM(ProSeq)
numpy_array = proteinVec.numpy()
np.save('protein_vec.npy', numpy_array)