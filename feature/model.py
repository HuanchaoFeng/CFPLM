from transformers import AutoTokenizer, AutoModelForMaskedLM,BertConfig
from torch.utils.data import DataLoader
import torch
import numpy as np
from Bio import SeqIO
import os
import esm

def read_protein(path):
    res = []
    rescords = list(SeqIO.parse(path, format="fasta"))
    for x in rescords:
        id = str(x.id)
        seq = str(x.seq)
        res.append((id, seq))
    return res

def ESM(OriginSeq):
    torch.cuda.empty_cache()
    EmbbingModel, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    EmbbingModel.to(device)
    EmbbingModel.eval() 
    batch_labels, batch_strs, batch_tokens = batch_converter(OriginSeq)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    dataloader = DataLoader(batch_tokens, batch_size=8)
    tmp = []

    print("dataloader:")
    for batch in dataloader:
        print(len(batch[0]))
        print(batch)
        # print(batch.shape)
        with torch.no_grad():
            torch.cuda.empty_cache()
            results = EmbbingModel(batch.to(device), repr_layers=[30], return_contacts=False)
            tmp.append(results["representations"][30])
    token_representations = torch.cat(tmp, dim=0)

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
    seq_tensor = torch.stack(sequence_representations)
    seq_tensor.to("cpu")
    return seq_tensor


from transformers import AutoTokenizer, AutoModelForMaskedLM,BertConfig
import torch
import numpy as np
from Bio import SeqIO
import os

def read_rna(path):
    res = []
    rescords = list(SeqIO.parse(path, format="fasta"))
    for x in rescords:
        id = str(x.id)
        seq = str(x.seq)
        res.append(seq) 
    return res

def BIRNA_BERT(OriginSeq):
    tokenizer = AutoTokenizer.from_pretrained("buetnlpbio/birna-tokenizer")
    config = BertConfig.from_pretrained("buetnlpbio/birna-bert")
    mysterybert = AutoModelForMaskedLM.from_pretrained("buetnlpbio/birna-bert", config=config, trust_remote_code=True)
    mysterybert.cls = torch.nn.Identity()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    mysterybert.to(device).eval()
    rna_sequences = OriginSeq
    all_embeddings = []

    for i in range(0, len(rna_sequences)):

        seq = rna_sequences[i]
        input = tokenizer(seq, return_tensors="pt", padding=True)
        input = {k: v.to(device) for k, v in input.items()}
        with torch.no_grad():  
            output = mysterybert(**input)
        embedding = output.logits[0, 0, :].detach().cpu()
        all_embeddings.append(embedding)

    all_embeddings_tensor = torch.stack(all_embeddings)
    all_embeddings_np = all_embeddings_tensor.numpy()
    np.save('rna_feature.npy', all_embeddings_np) 

RnaSeq=read_rna('rna.fasta')
BIRNA_BERT(RnaSeq)




ProSeq=read_protein(path)
proteinVec=ESM(ProSeq)
numpy_array = proteinVec.numpy()
np.save('protein_vec.npy', numpy_array)