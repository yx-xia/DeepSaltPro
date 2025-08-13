from transformers import AutoTokenizer, AutoModelForTextEncoding
from Bio.SeqIO import parse
import torch
import time
import gc
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))


def get_Ankh_model():
    model = AutoModelForTextEncoding.from_pretrained('./model/ankh')
    model = model.to(device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained('./model/ankh', do_lower_case=False)
    return model, tokenizer


def get_embeddings(model, tokenizer, seqs, per_residue, per_protein, sec_struct,
                   max_residues=4000, max_seq_len=3000, max_batch=1):
    res_names = []
    if sec_struct:
        pass
    results = {"residue_embs": dict(),
               "protein_embs": dict(),
               "sec_structs": dict()
               }
    seq_dict = sorted(seqs.items(), key=lambda kv: len(seqs[kv[0]]), reverse=True)
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            print(pdb_ids)
            res_names.append(pdb_ids)
            print(seq_lens)
            batch = list()
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]
                if per_residue:
                    results["residue_embs"][identifier] = emb.detach().cpu().numpy().squeeze()
                if per_protein:
                    protein_emb = emb.mean(dim=0)
                    results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()

    return results, res_names


def format(file_path):
    sequences = []
    labels = []
    for record in parse(file_path, "fasta"):
        seq = str(record.seq)
        if len(seq) > 7000:
            seq = seq[:7000]
        sequences.append(seq)
        labels.append(0)
    seq_dict = dict(zip(range(len(sequences)), sequences))
    return seq_dict


if __name__ == '__main__':
    model, tokenizer = get_Ankh_model()
    gc.collect()
    per_residue = 0
    per_protein = 1
    sec_struct = 0

    file_paths = [
        './dataset/train_P.fasta',
        './dataset/train_N.fasta',
        './dataset/test_P.fasta',
        './dataset/test_N.fasta'
    ]

    for file_path in file_paths:
        seq_dict = format(file_path)
        results, res_names = get_embeddings(model, tokenizer, seq_dict,
                                            per_residue, per_protein, sec_struct)
        protein_embs = [torch.tensor(emb) for emb in results["protein_embs"].values()]
        embeddings = torch.stack(protein_embs)

        file_name = file_path.split('/')[-1].replace('.fasta', '')
        save_path = f'./Pre-trained_features/Ankh/{file_name}_to_Ankh.pt'
        torch.save(embeddings, save_path)
        print(f'{save_path} saved successfully.')
    