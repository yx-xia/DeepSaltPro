import torch.nn as nn
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE as SMOTE_imb
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN as ADASYN_imb
from imblearn.over_sampling import ADASYN as ADASYN_imb
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN as SMOTEENN_imb

def SMOTEENN(x_train, y_train, smote_k_neighbors=5, enn_n_neighbors=3, enn_kind='all', seed=42):
    """
    使用imblearn库的SMOTEENN处理不平衡数据集
    输入：
    x_train, y_train: 交叉验证划分后的、张量类型的训练集特征和标签
    smote_k_neighbors: SMOTE的最近邻数，用于生成新样本
    enn_n_neighbors: ENN的最近邻数，用于移除边界样本
    enn_kind: ENN的策略 ('all' 更激进, 'mode' 更保守)
    seed: 随机种子用于可复现性

    返回：
    x_train_new, y_train_new: 处理后的、张量类型的训练集特征和标签
    """
    print(f'SMOTEENN, smote_k_neighbors={smote_k_neighbors}, enn_n_neighbors={enn_n_neighbors}, enn_kind={enn_kind}')
    # 创建SMOTEENN实例
    smoteenn = SMOTEENN_imb(random_state=seed)

    # 将PyTorch张量转换为NumPy数组
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    # 使用SMOTEENN进行数据处理
    x_train_res, y_train_res = smoteenn.fit_resample(x_train_np, y_train_np)

    # 将NumPy数组转换回PyTorch张量
    x_train_new = torch.from_numpy(x_train_res).float()
    y_train_new = torch.from_numpy(y_train_res).float()

    # 同时打乱数据和标签
    x_train_new_np = x_train_new.cpu().numpy()
    y_train_new_np = y_train_new.cpu().numpy()
    x_train_new_np, y_train_new_np = shuffle(x_train_new_np, y_train_new_np, random_state=123)
    x_train_new = torch.from_numpy(x_train_new_np)
    y_train_new = torch.from_numpy(y_train_new_np)

    return x_train_new, y_train_new
def ENN(x_train, y_train, n_neighbors=3, kind = 'all', seed=42):
    """
    使用imblearn库的ENN（Edited Nearest Neighbours）处理不平衡数据集
    输入：
    x_train, y_train: 交叉验证划分后的、张量类型的训练集特征和标签
    seed: 随机种子用于可复现性
    n_neighbors: 最近邻的数量，用于编辑数据集
    kind_sel: all 更激进, mode 更保守

    返回：
    x_train_new, y_train_new: 下采样后的、张量类型的训练集特征和标签
    """
    if kind == 'all':
        print(f'ENN, n_neighbors={n_neighbors}, kind=all')
        enn = EditedNearestNeighbours(
        sampling_strategy="auto",  # 对多数类进行平衡
        n_neighbors=n_neighbors,    # 最近邻数量
        kind_sel="all",            # 移除所有不一致样本
        )
    elif kind == 'mode':
        print(f'ENN, n_neighbors={n_neighbors}, kind=mode')
        enn = EditedNearestNeighbours(
        sampling_strategy="auto",  # 对多数类进行平衡
        n_neighbors=n_neighbors,    # 最近邻数量
        kind_sel="mode",            # 移除所有不一致样本
        )

    # 将PyTorch张量转换为NumPy数组
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    # 使用ENN进行下采样
    x_train_res, y_train_res = enn.fit_resample(x_train_np, y_train_np)

    # 将NumPy数组转换回PyTorch张量
    x_train_new = torch.from_numpy(x_train_res).float()
    y_train_new = torch.from_numpy(y_train_res).float()

    # 同时打乱数据和标签
    x_train_new_np = x_train_new.cpu().numpy()
    y_train_new_np = y_train_new.cpu().numpy()
    x_train_new_np, y_train_new_np = shuffle(x_train_new_np, y_train_new_np, random_state=123)
    x_train_new = torch.from_numpy(x_train_new_np)
    y_train_new = torch.from_numpy(y_train_new_np)

    return x_train_new, y_train_new

def ADASYN(x_train, y_train, seed=42):
    """
    使用imblearn库的ADASYN过采样处理不平衡数据集
    输入：
    x_train, y_train: 交叉验证划分后的、张量类型的训练集特征和标签
    seed: 随机种子用于可复现性

    返回：
    x_train_new, y_train_new: 过采样后的、张量类型的训练集特征和标签
    """
    print('ADASYN')
    # 设置随机种子以确保结果的复现性
    adasyn = ADASYN_imb(random_state=seed)

    # 将PyTorch张量转换为NumPy数组
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    # 使用ADASYN进行过采样
    x_train_res, y_train_res = adasyn.fit_resample(x_train_np, y_train_np)

    # 将NumPy数组转换回PyTorch张量
    x_train_new = torch.from_numpy(x_train_res).float()
    y_train_new = torch.from_numpy(y_train_res).float()

    # 同时打乱数据和标签
    x_train_new_np = x_train_new.cpu().numpy()
    y_train_new_np = y_train_new.cpu().numpy()
    x_train_new_np, y_train_new_np = shuffle(x_train_new_np, y_train_new_np, random_state=123)
    x_train_new = torch.from_numpy(x_train_new_np)
    y_train_new = torch.from_numpy(y_train_new_np)

    return x_train_new, y_train_new

def ROS(x_train, y_train, seed=42):
    """
    处理不平衡数据集：随机上采样
    输入：
    x_train, y_train: 交叉验证划分后的、张量类型的训练集特征和标签
    seed: 随机种子用于可复现性

    返回：
    x_train_new, y_train_new: 上采样后的、张量类型的训练集特征和标签
    """
    print('ROS')
    # 设置随机种子以确保代码的复现性
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 计算正负样本的索引
    positive_indices = torch.where(y_train == 1)[0]
    negative_indices = torch.where(y_train == 0)[0]

    # 确保少数类和多数类样本数量相等
    num_positive = positive_indices.size(0)
    num_negative = negative_indices.size(0)

    # 随机上采样少数类
    if num_positive < num_negative:
        oversampled_positive_indices = positive_indices[
            torch.randint(0, num_positive, (num_negative - num_positive,), generator=torch.manual_seed(seed))
        ]
        new_indices = torch.cat([positive_indices, oversampled_positive_indices, negative_indices])
    else:
        # 如果正样本多于负样本，随机上采样负样本
        oversampled_negative_indices = negative_indices[
            torch.randint(0, num_negative, (num_positive - num_negative,), generator=torch.manual_seed(seed))
        ]
        new_indices = torch.cat([negative_indices, oversampled_negative_indices, positive_indices])

    # 根据新索引重新组织训练集的特征和标签
    x_train_new = x_train[new_indices]
    y_train_new = y_train[new_indices]

    # 同时打乱数据和标签
    x_train_new_np = x_train_new.cpu().numpy()
    y_train_new_np = y_train_new.cpu().numpy()
    x_train_new_np, y_train_new_np = shuffle(x_train_new_np, y_train_new_np, random_state=123)
    x_train_new = torch.from_numpy(x_train_new_np)
    y_train_new = torch.from_numpy(y_train_new_np)

    return x_train_new, y_train_new
def Borderline_SMOTE(x_train, y_train, kind=1, seed=42):
    """
    使用imblearn库的Borderline-SMOTE过采样处理不平衡数据集
    输入：
    x_train, y_train: 交叉验证划分后的、张量类型的训练集特征和标签
    seed: 随机种子用于可复现性

    参数 kind:
    borderline-1: 仅从少数类样本和其少数类近邻之间生成合成样本, 如果您更专注于提升少数类样本在边界附近的分布密度，这种方法更安全且效果良好。
    borderline-2: 从少数类样本和其少数类近邻之间生成样本，同时也可能从少数类和多数类近邻之间生成样本, 如果希望增强模型的鲁棒性，即使边界附近存在一定的不确定性，这种方法适合生成更多边界附近的多样性样本。

    返回：
    x_train_new, y_train_new: 过采样后的、张量类型的训练集特征和标签
    """

    # 设置随机种子以确保结果的复现性
    if kind == 1:
        print('Borderline-SMOTE, kind=borderline-1')
        smote = BorderlineSMOTE(random_state=seed, kind="borderline-1")
    elif kind == 2:
        print('Borderline-SMOTE, kind=borderline-2')
        smote = BorderlineSMOTE(random_state=seed, kind="borderline-2")

    # 将PyTorch张量转换为NumPy数组
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    # 使用Borderline-SMOTE进行过采样
    x_train_res, y_train_res = smote.fit_resample(x_train_np, y_train_np)

    # 将NumPy数组转换回PyTorch张量
    x_train_new = torch.from_numpy(x_train_res).float()
    y_train_new = torch.from_numpy(y_train_res).float()

    # 同时打乱数据和标签
    x_train_new_np = x_train_new.cpu().numpy()
    y_train_new_np = y_train_new.cpu().numpy()
    x_train_new_np, y_train_new_np = shuffle(x_train_new_np, y_train_new_np, random_state=123)
    x_train_new = torch.from_numpy(x_train_new_np)
    y_train_new = torch.from_numpy(y_train_new_np)

    return x_train_new, y_train_new

def SMOTE(x_train, y_train, seed=42):
    """
    使用imblearn库的SMOTE过采样处理不平衡数据集
    输入：
    x_train, y_train: 交叉验证划分后的、张量类型的训练集特征和标签
    seed: 随机种子用于可复现性

    返回：
    x_train_new, y_train_new: 过采样后的、张量类型的训练集特征和标签
    """
    print('SMOTE')
    # 设置随机种子以确保结果的复现性
    smote = SMOTE_imb(random_state=seed)

    # 将PyTorch张量转换为NumPy数组
    x_train_np = x_train.cpu().numpy()
    y_train_np = y_train.cpu().numpy()

    # 使用SMOTE进行过采样
    x_train_res, y_train_res = smote.fit_resample(x_train_np, y_train_np)

    # 将NumPy数组转换回PyTorch张量
    x_train_new = torch.from_numpy(x_train_res).float()
    y_train_new = torch.from_numpy(y_train_res).float()

    # 同时打乱数据和标签
    x_train_new_np = x_train_new.cpu().numpy()
    y_train_new_np = y_train_new.cpu().numpy()
    x_train_new_np, y_train_new_np = shuffle(x_train_new_np, y_train_new_np, random_state=123)
    x_train_new = torch.from_numpy(x_train_new_np)
    y_train_new = torch.from_numpy(y_train_new_np)

    return x_train_new, y_train_new

def RUS(x_train, y_train, seed=42):
    """
    处理不平衡数据集：随机下采样
    输入：
    x_train, y_train: 交叉验证划分后的、张量类型的训练集特征和标签
    seed: 随机种子用于可复现性

    返回：
    x_train_new, y_train_new: 过采样后的、张量类型的训练集特征和标签
    """
    print('RUS')
    # 设置随机种子以确保代码的复现性
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 将标签张量转换为numpy数组，便于处理
    y_train_np = y_train.numpy()

    # 计算正负样本的索引
    positive_indices = torch.where(y_train == 1)[0]
    negative_indices = torch.where(y_train == 0)[0]

    # 确保正样本数量不超过负样本数量
    num_positive = positive_indices.size(0)
    num_negative = negative_indices.size(0)

    # 如果负样本多于正样本，则从负样本中随机选择等量于正样本的负样本
    if num_negative > num_positive:
        reduced_negative_indices = negative_indices[torch.randperm(negative_indices.size(0))[:num_positive]]
        new_indices = torch.cat([positive_indices, reduced_negative_indices])
    else:
        new_indices = torch.cat([positive_indices, negative_indices])

    # 根据新索引重新组织训练集的特征和标签
    x_train_new = x_train[new_indices]
    y_train_new = y_train[new_indices]

    # 同时打乱数据和标签
    x_train_new_np = x_train_new.cpu().numpy()
    y_train_new_np = y_train_new.cpu().numpy()
    x_train_new_np, y_train_new_np = shuffle(x_train_new_np, y_train_new_np, random_state=123)
    x_train_new = torch.from_numpy(x_train_new_np)
    y_train_new = torch.from_numpy(y_train_new_np)

    return x_train_new, y_train_new

class Focal_Loss(torch.nn.Module):
    """
    焦点损失函数，参数：alpha，gamma
    """
    def __init__(self, alpha=0.5, gamma=0):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary classification
    Args:
        alpha (float): Weight of False Positives (FP). Default: 0.5
        beta (float): Weight of False Negatives (FN). Default: 0.5
        smooth (float): Smoothing factor to avoid division by zero. Default: 1e-6
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted probabilities (output of sigmoid), shape (N, *).
            targets: Ground truth labels, shape (N, *).
        Returns:
            Tversky Loss value (scalar).
        """
        # Flatten inputs and targets for easier computation
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
        TP = (inputs * targets).sum()  # True Positive
        FP = ((1 - targets) * inputs).sum()  # False Positive
        FN = (targets * (1 - inputs)).sum()  # False Negative

        # Compute Tversky index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Compute Tversky Loss
        tversky_loss = 1 - tversky_index

        return tversky_loss

class Weighted_Cross_Entropy_Loss(torch.nn.Module):
    def __init__(self, pos_weight=1, neg_weight=1):
        super(Weighted_Cross_Entropy_Loss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
    def forward(self, probs, target):
        # 计算加权损失
        loss = - (self.pos_weight * target * torch.log(probs + 1e-9) +
                  self.neg_weight * (1 - target) * torch.log(1 - probs + 1e-9))
        return loss.mean()

class EarlyStopping:
    """
    早停，参数：patience，delta
    patience：当验证损失连续 patience 个周期没有改善时，将触发早停
    delta：如果验证损失的改善 < delta，则不认为是有效的改善
    """
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience    # 当验证损失连续 patience 个周期没有改善时，将触发早停。
        self.delta = delta  # 如果验证损失的改善<delta，则不认为是有效的改善
        self.best_loss = np.Inf # 最优 loss 初始为无穷大
        self.counter = 0    # 当前没有改善的周期数
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# 定义对比损失函数类
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算欧几里得距离
        euclidean_distance = F.pairwise_distance(output1, output2)

        # 计算对比损失
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +  # 正样本
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # 负样本
        )

        return loss_contrastive

def collate(batch):
    device = torch.device("cpu")
    # 初始化存储不同数据的列表
    seq1_ls = []  # 存储第一个序列特征的列表
    seq2_ls = []  # 存储第二个序列特征的列表
    label1_ls = []  # 存储第一个序列标签的列表
    label2_ls = []  # 存储第二个序列标签的列表
    label_ls = []  # 存储二进制标签的列表

    # 获取 batch 的大小
    batch_size = len(batch)

    # 遍历 batch 的一半，生成样本对
    for i in range(int(batch_size / 2)):
        # 获取第 i 个样本的序列特征、标签和原始序列
        seq1, label1 = batch[i][0], batch[i][1]
        # 获取第 i + batch_size / 2 个样本的序列特征、标签和原始序列
        seq2, label2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1]

        # 将第一个样本的标签添加到列表中，并扩展维度
        label1_ls.append(label1.unsqueeze(0))
        # 将第二个样本的标签添加到列表中，并扩展维度
        label2_ls.append(label2.unsqueeze(0))

        # 计算二进制标签，若两个标签不同则为负样本对（1），相同则为正样本对（0）
        if label1 == label2:
            label = 0
        else:
            label = 1

        # 将第一个样本的序列特征添加到列表中，并扩展维度
        seq1_ls.append(seq1.unsqueeze(0))
        # 将第二个样本的序列特征添加到列表中，并扩展维度
        seq2_ls.append(seq2.unsqueeze(0))

        # 将二进制标签添加到列表中，并扩展维度
        label_ls.append(torch.tensor(label).unsqueeze(0))

    # 将列表中的张量合并为一个大的张量，并移动到设备上
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)

    # 返回合并后的张量和原始序列列表
    return seq1, seq2, label, label1, label2

class MyDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def caculate_metric(pred_y, labels, pred_prob):
    """
    计算各种评估指标，包括准确率 (ACC)、精确率 (Precision)、召回率 (Recall/Sensitivity)、特异性 (Specificity)、F1 分数、AUC（曲线下面积）、MCC（马修斯相关系数）等。
    还计算 ROC 曲线和 PR 曲线的数据。

    参数：
    pred_y (list or array): 模型预测的类别标签
    labels (list or array): 真实的标签
    pred_prob (list or array): 模型预测的概率值

    返回：
    metric (torch.Tensor): 计算得到的评估指标，包括 ACC、Precision、Recall、Specificity、F1、AUC 和 MCC
    roc_data (list): ROC 曲线的数据，包括假阳性率 (fpr)、真正率 (tpr) 和 AUC
    prc_data (list): PR 曲线的数据，包括召回率 (recall)、精确率 (precision) 和 AP（平均精度）
    """

    test_num = len(labels)  # 测试样本的数量
    tp = 0  # 真阳性数量
    fp = 0  # 假阳性数量
    tn = 0  # 真阴性数量
    fn = 0  # 假阴性数量

    # 遍历每个样本，计算 TP, FP, TN, FN
    for index in range(test_num):
        if int(labels[index]) == 1:  # 真实标签为正样本
            if labels[index] == pred_y[index]:  # 预测结果也为正样本
                tp = tp + 1  # 正确的正样本预测
            else:
                fn = fn + 1  # 错误的正样本预测
        else:  # 真实标签为负样本
            if labels[index] == pred_y[index]:  # 预测结果也为负样本
                tn = tn + 1  # 正确的负样本预测
            else:
                fp = fp + 1  # 错误的负样本预测

    # 计算准确率 (Accuracy)
    ACC = float(tp + tn) / test_num

    # 计算精确率 (Precision)
    if tp + fp == 0:
        Precision = 0  # 防止除以零
    else:
        Precision = float(tp) / (tp + fp)

    # 计算召回率 (Recall/Sensitivity)
    if tp + fn == 0:
        Recall = Sensitivity = 0  # 防止除以零
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # 计算特异性 (Specificity)
    if tn + fp == 0:
        Specificity = 0  # 防止除以零
    else:
        Specificity = float(tn) / (tn + fp)

    # 计算马修斯相关系数 (MCC)
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0  # 防止除以零
    else:
        # MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        MCC = float(tp * tn - fp * fn) / np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))

    # 计算 F1 分数
    if Recall + Precision == 0:
        F1 = 0  # 防止除以零
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # 计算 ROC 曲线数据
    labels = list(map(int, labels))  # 转换标签为整数
    pred_prob = list(map(float, pred_prob))  # 转换预测概率为浮点数
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 计算假阳性率、真正率和阈值
    AUC = auc(fpr, tpr)  # 计算 ROC 曲线下面积

    # 计算 PR 曲线数据
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)  # 计算精确率、召回率和阈值
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)  # 计算平均精度

    # 返回计算得到的指标和曲线数据
    metric = [ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC]
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data
