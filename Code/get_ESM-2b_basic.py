import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
import esm
import gc

# CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('GPU available')
else:
    device = torch.device('cpu')
    print('CPU available')

# 加载模型和字母表
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()

def esm_to_feature(tuple_seq_list, num_layer, device):
    i = 1
    embeddings = []
    for tuple_seq in tuple_seq_list:
        # print(f"{i}/{len(tuple_seq_list)}")
        i += 1
        print('长度: ', len(tuple_seq[1]))
        print(tuple_seq[1]) # tuple_seq是一个tuple
        # 将蛋白质序列列表加载到批处理转换器中，且批处理转换器接收的是一个list变量
        batch_labels, batch_strs, batch_tokens = batch_converter([tuple_seq])
        batch_tokens = batch_tokens.to(device)  # 将批处理令牌移动到所选设备上

        # 提取每个残基的表示
        with torch.no_grad():
            # repr_layers 等于模型层数
            results = model(batch_tokens, repr_layers=[num_layer], return_contacts=False)
        token_representations = results["representations"][num_layer]

        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  # 记录每条序列的真实长度+2 (开始和结束标记符)
        # 通过平均值生成每个序列的表示
        # 注意：令牌0始终是序列开始令牌，所以第一个残基的令牌是1。
        sequence_representation = token_representations[0, 1:batch_lens[0] - 1].mean(0)
        embeddings.append(sequence_representation)

        del batch_tokens, results, token_representations, batch_labels, batch_strs
        torch.cuda.empty_cache()  # 清理 GPU 缓存
        gc.collect()  # 释放 Python 内存

    return torch.stack(embeddings)

def format(file_path):
    # 初始化两个空列表，用于存储处理后的序列和对应的标签
    sequences = []
    labels = []
    # 使用 SeqIO.parse 函数解析 FASTA 文件，每次迭代取出一个序列记录
    for record in SeqIO.parse(file_path, "fasta"):
        # 将记录中的序列部分转换为字符串
        seq = str(record.seq)
         # 检查序列的长度，如果长度超过 7000，则截取前 7000 个字符
        if len(seq) > 7000:
            seq = seq[:7000]
        # 将一个空字符串和处理后的序列组成一个元组，添加到 sequences 列表中
        sequences.append(tuple(['', seq]))
        # 为当前序列添加一个标签 0 到 labels 列表中
        labels.append(0)
    # 函数执行完毕后，返回存储序列的列表和存储标签的列表
    return sequences, labels

file_path = './dataset/train_P.fasta'
tuple_seq_list, _ = format(file_path)
embeddings = esm_to_feature(tuple_seq_list, num_layer=36, device=device)
torch.save(embeddings, './Pre-trained_features/ESM-2b/train_P_to_ESM-2b.pt')
print('done')

file_path = './dataset/train_N.fasta'
tuple_seq_list, _ = format(file_path)
embeddings = esm_to_feature(tuple_seq_list, num_layer=36, device=device)
torch.save(embeddings, './Pre-trained_features/ESM-2b/train_N_to_ESM-2b.pt')
print('done')

file_path = './dataset/test_P.fasta'
tuple_seq_list, _ = format(file_path)
embeddings = esm_to_feature(tuple_seq_list, num_layer=36, device=device)
torch.save(embeddings, './Pre-trained_features/ESM-2b/test_P_to_ESM-2b.pt')
print('done')

file_path = './dataset/test_N.fasta'
tuple_seq_list, _ = format(file_path)
embeddings = esm_to_feature(tuple_seq_list, num_layer=36, device=device)
torch.save(embeddings, './Pre-trained_features/ESM-2b/test_N_to_ESM-2b.pt')
print('done')