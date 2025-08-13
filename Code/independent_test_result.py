import pickle
import numpy as np
import pandas as pd
import torch
import os
import random
import math
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix, silhouette_score, auc
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.patches as patches
from KANLayer import KANLayer
from utils import EarlyStopping, caculate_metric
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

# 设置 GPU
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 固定随机种子
seed_n = 2
print(f'Seed: {seed_n}')
g = torch.Generator()
g.manual_seed(seed_n)
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['PYTHONHASHSEED'] = str(seed_n)

def get_train_test_data():
    with open("./Pre-trained_features/ESM-2b/train_P_to_ESM-2b.pt", "rb") as tf:
        feature_dict = torch.load(tf, map_location=torch.device('cpu'))
        train_ESM2_P = feature_dict.float().numpy()
    with open("./Pre-trained_features/ESM-2b/train_N_to_ESM-2b.pt", "rb") as tf:
        feature_dict = torch.load(tf, map_location=torch.device('cpu'))
        train_ESM2_N = feature_dict.float().numpy()
    train_ESM2 = np.vstack((train_ESM2_P, train_ESM2_N))

    with open("./Pre-trained_features/Ankh/train_P_to_Ankh.pt", "rb") as f:
        feature_dict = torch.load(f, map_location=torch.device('cpu'))
        train_Ankh_P = feature_dict.numpy()
    with open("./Pre-trained_features/Ankh/train_N_to_Ankh.pt", "rb") as f:
        feature_dict = torch.load(f, map_location=torch.device('cpu'))
        train_Ankh_N = feature_dict.numpy()
    train_Ankh = np.vstack((train_Ankh_P, train_Ankh_N))

    num_pos = train_ESM2_P.shape[0]
    num_neg = train_ESM2_N.shape[0]
    train_labels = np.hstack(([1] * num_pos, [0] * num_neg))

    with open("./Pre-trained_features/ESM-2b/test_P_to_ESM-2b.pt", "rb") as tf:
        test_feature_dict_P = torch.load(tf)
        test_ESM2_P = test_feature_dict_P.to(torch.float32).cpu().numpy()
    with open("./Pre-trained_features/ESM-2b/test_N_to_ESM-2b.pt", "rb") as tf:
        test_feature_dict_N = torch.load(tf)
        test_ESM2_N = test_feature_dict_N.to(torch.float32).cpu().numpy()
    test_ESM2 = np.vstack((test_ESM2_P, test_ESM2_N))

    with open("./Pre-trained_features/Ankh/test_P_to_Ankh.pt", "rb") as f:
        feature_dict = torch.load(f, map_location=torch.device('cpu'))
        test_Ankh_P = feature_dict.numpy()
    with open("./Pre-trained_features/Ankh/test_N_to_Ankh.pt", "rb") as f:
        feature_dict = torch.load(f, map_location=torch.device('cpu'))
        test_Ankh_N = feature_dict.numpy()
    test_Ankh = np.vstack((test_Ankh_P, test_Ankh_N))

    num_pos_test = test_ESM2_P.shape[0]
    num_neg_test = test_ESM2_N.shape[0]
    test_labels = np.hstack(([1] * num_pos_test, [0] * num_neg_test))

    scaler_esm2 = MinMaxScaler()
    scaler_ankh = MinMaxScaler()
    pca_esm2 = PCA(n_components=512, random_state=seed_n)
    pca_ankh = PCA(n_components=512, random_state=seed_n)

    X_train_esm2 = scaler_esm2.fit_transform(train_ESM2)
    X_test_esm2 = scaler_esm2.transform(test_ESM2)
    X_train_ankh = scaler_ankh.fit_transform(train_Ankh)
    X_test_ankh = scaler_ankh.transform(test_Ankh)

    X_train_esm2 = pca_esm2.fit_transform(X_train_esm2)
    X_test_esm2 = pca_esm2.transform(X_test_esm2)
    X_train_ankh = pca_ankh.fit_transform(X_train_ankh)
    X_test_ankh = pca_ankh.transform(X_test_ankh)
   
    print("ESM-2 variance retained:", np.sum(pca_esm2.explained_variance_ratio_))
    print("Ankh variance retained:", np.sum(pca_ankh.explained_variance_ratio_))  
    print(f"训练集: {sum(train_labels == 1)} 正样本, {sum(train_labels == 0)} 负样本")

    return (X_train_esm2, X_train_ankh), (X_test_esm2, X_test_ankh), train_labels, test_labels, scaler_esm2, scaler_ankh, pca_esm2, pca_ankh

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.esm2_cnn1 = nn.Sequential(
            nn.Conv1d(512, 256, 9, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 9, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.esm2_cnn_gru = nn.Sequential(
            nn.Conv1d(512, 512, 9, padding='same'),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.ankh_cnn1 = nn.Sequential(
            nn.Conv1d(512, 256, 9, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 9, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.ankh_cnn_gru = nn.Sequential(
            nn.Conv1d(512, 512, 9, padding='same'),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.bigru = nn.GRU(
            input_size=512+512,
            hidden_size=64,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.4)
        self.kan1 = KANLayer(
            in_dim=128+128+128,  
            out_dim=16,
            num=5,
            k=3,
            grid_eps=0.5,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            noise_scale=0.5,
            device=device
        )
        self.kan2 = KANLayer(
            in_dim=16,
            out_dim=1,
            num=3,
            k=3,
            grid_eps=0.5,
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            noise_scale=0.2,
            device=device
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, esm2_x, ankh_x, return_intermediate=False):
        esm2 = esm2_x
        ankh = ankh_x

        # CNN 处理 ESM2 和 Ankh 输入
        esm2_cnn_1 = self.esm2_cnn1(esm2).permute(0, 2, 1)  # [batch_size, 1, 128]
        ankh_cnn_1 = self.ankh_cnn1(ankh).permute(0, 2, 1)  # [batch_size, 1, 128]
        esm2_cnn_gru = self.esm2_cnn_gru(esm2).permute(0, 2, 1)  # [batch_size, 1, 512]
        ankh_cnn_gru = self.ankh_cnn_gru(ankh).permute(0, 2, 1)  # [batch_size, 1, 512]

        # GRU 输入拼接
        gru_inputs = torch.concat([esm2_cnn_gru, ankh_cnn_gru], dim=-1)  # [batch_size, 1, 512+512]
        gru_out, _ = self.bigru(gru_inputs)  # [batch_size, 1, 128]

        # 拼接 CNN 和 GRU 输出
        concat_out = torch.concat([esm2_cnn_1, gru_out, ankh_cnn_1], dim=-1)  # [batch_size, 1, 128+128+128]
        concat_out = concat_out.squeeze(1)  # [batch_size, 384]

        # KAN 层处理
        x_out, preacts1, postacts1, postspline1 = self.kan1(concat_out)
        kan1_out = x_out
        x_out, preacts2, postacts2, postspline2 = self.kan2(x_out)
        x_out = x_out.squeeze(-1)  # [batch_size]

        if return_intermediate:
            return x_out, esm2_cnn_gru.squeeze(1), ankh_cnn_gru.squeeze(1), gru_out.squeeze(1), kan1_out
        return x_out

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=128):
    X_train_esm2, X_train_ankh = X_train
    X_val_esm2, X_val_ankh = X_val

    X_train_esm2_tensor = torch.FloatTensor(X_train_esm2).unsqueeze(2).to(device)
    X_train_ankh_tensor = torch.FloatTensor(X_train_ankh).unsqueeze(2).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    
    X_val_esm2_tensor = torch.FloatTensor(X_val_esm2).unsqueeze(2).to(device)
    X_val_ankh_tensor = torch.FloatTensor(X_val_ankh).unsqueeze(2).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    print(f"X_train_esm2_tensor shape: {X_train_esm2_tensor.shape}")
    print(f"X_train_ankh_tensor shape: {X_train_ankh_tensor.shape}")

    train_dataset = TensorDataset(X_train_esm2_tensor, X_train_ankh_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=4e-5, weight_decay=1e-4)  
    criterion = FocalLoss(alpha=0.5, gamma=1.0)  # 使用 Focal Loss
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=4e-7)
    early_stopping = EarlyStopping(patience=25, delta=0.005)

    best_val_mcc = -1.0
    best_model_state = None
    model_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_mcc': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for esm2_inputs, ankh_inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(esm2_inputs, ankh_inputs)
            loss = criterion(outputs, targets)
            predicted = outputs > 0.0
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        model_history['train_loss'].append(train_loss / len(train_loader))
        model_history['train_acc'].append(train_acc)

        model.eval()
        with torch.no_grad():
            outputs = model(X_val_esm2_tensor, X_val_ankh_tensor)
            val_loss = criterion(outputs, y_val_tensor).item()
            predicted = outputs > 0.0
            predicted_class = predicted.cpu().numpy()
            predicted_prob = torch.sigmoid(outputs).cpu().numpy()
            metric, _, _ = caculate_metric(predicted_class, y_val_tensor.cpu().numpy(), predicted_prob)
            val_mcc = metric[6]

        model_history['val_loss'].append(val_loss)
        model_history['val_mcc'].append(val_mcc)
        model_history['lr'].append(optimizer.param_groups[0]['lr'])

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state = model.state_dict().copy()

        if early_stopping(-val_mcc):  # 使用负 MCC 以最小化
            print(f"Early stopping at epoch {epoch + 1}")
            break

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_val_mcc, model_history

def train_five_fold_CV(X_train, y_train):
    X_train_esm2, X_train_ankh = X_train
    kf = KFold(n_splits=5, shuffle=True, random_state=seed_n)
    models = []
    metrics_collection = {
        'ACC': [], 'SP': [], 'SN': [], 'PRE': [], 'F1': [], 'MCC': [], 'AUC': [], 'AUPR': []
    }
    best_val_mcc_collection = []

    # 新增：收集每折的 ROC 和 PR 曲线数据
    fprs, tprs, precisions, recalls = [], [], [], []
   
    for fold, (train_index, test_index) in enumerate(kf.split(X_train_esm2)):
        print(f"\nFold {fold + 1}/5")
        X_train_CV_esm2, X_valid_CV_esm2 = X_train_esm2[train_index], X_train_esm2[test_index]
        X_train_CV_ankh, X_valid_CV_ankh = X_train_ankh[train_index], X_train_ankh[test_index]
        y_train_CV, y_valid_CV = y_train[train_index], y_train[test_index]

        model, fold_mcc, model_history = train_model(
            (X_train_CV_esm2, X_train_CV_ankh),
            y_train_CV,
            (X_valid_CV_esm2, X_valid_CV_ankh),
            y_valid_CV
        )

        model.eval()
        with torch.no_grad():
            X_valid_esm2_tensor = torch.FloatTensor(X_valid_CV_esm2).unsqueeze(2).to(device)
            X_valid_ankh_tensor = torch.FloatTensor(X_valid_CV_ankh).unsqueeze(2).to(device)
            outputs = model(X_valid_esm2_tensor, X_valid_ankh_tensor)
            predicted_probability = torch.sigmoid(outputs).cpu().numpy()
            predicted_class = (predicted_probability > 0.5).astype(int)

        metric, roc_data, prc_data = caculate_metric(predicted_class, y_valid_CV, predicted_probability)
        metrics = {
            'ACC': metric[0],
            'PRE': metric[1],
            'SN': metric[2],
            'SP': metric[3],
            'F1': metric[4],
            'AUC':metric[5],
            'MCC': metric[6],
            'AUPR': prc_data[2]
        }

        # 收集 ROC 和 PR 曲线数据
        fprs.append(roc_data[0])  # FPR
        tprs.append(roc_data[1])  # TPR
        precisions.append(prc_data[1])  # Precision
        recalls.append(prc_data[0])  # Recall
        
        print(f"Fold {fold + 1} Evaluation Results:")
        print(f"Accuracy: {metrics['ACC']:.4f}, Specificity: {metrics['SP']:.4f}, Sensitivity: {metrics['SN']:.4f}")
        print(f"Precision: {metrics['PRE']:.4f}, F1 Score: {metrics['F1']:.4f}, MCC: {metrics['MCC']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}, AUPR: {metrics['AUPR']:.4f}")

        for key in metrics_collection:
            metrics_collection[key].append(metrics[key])
        best_val_mcc_collection.append(fold_mcc)

        fold_model_path = f'./save/model/fold_{fold+1}_ankh-esm2-net-kan.pt'
        torch.save(model.state_dict(), fold_model_path)
        print(f"Model saved to {fold_model_path}")

        models.append(model)

    # 计算五折平均 ROC 和 PR 曲线数据
    # 确保每折的 FPR 和 Recall 点数对齐（通过插值）
    n_points = 100  # 统一插值点数
    mean_fpr = np.linspace(0, 1, n_points)
    mean_recall = np.linspace(0, 1, n_points)
    interp_tprs = []
    interp_precisions = []

    for fpr, tpr, recall, precision in zip(fprs, tprs, recalls, precisions):
        # 对 ROC 曲线插值
        interp_tpr = interp1d(fpr, tpr, bounds_error=False, fill_value=(tpr[0], tpr[-1]))
        interp_tprs.append(interp_tpr(mean_fpr))
        # 对 PR 曲线插值
        interp_precision = interp1d(recall, precision, bounds_error=False, fill_value=(precision[0], precision[-1]))
        interp_precisions.append(interp_precision(mean_recall))

    # 计算平均值
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_precision = np.mean(interp_precisions, axis=0)
    # 计算平均 AUC 和 AP
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_ap = auc(mean_recall, mean_precision)

    # 保存为 CSV，包含 AP 和 AUC
    pr_curve_data = pd.DataFrame({
        'Precision': mean_precision,
        'Recall': mean_recall,
        'AP': [mean_ap] * n_points  # 重复 AP 值以匹配曲线点数
    })
    roc_curve_data = pd.DataFrame({
        'FPR': mean_fpr,
        'TPR': mean_tpr,
        'AUC': [mean_auc] * n_points  # 重复 AUC 值以匹配曲线点数
    })

    # 保存为 CSV
    pr_curve_data.to_csv('./save/Curve/pr_data_ankh-esm2-net-kan.csv', index=False)
    roc_curve_data.to_csv('./save/Curve/roc_data_ankh-esm2-net-kan.csv', index=False)
    print("Saved average PR and ROC curve data to ./save/Curve/")
    
    print("\nCross-Validation Results:")
    for key in metrics_collection:
        print(f"{key}: {np.mean(metrics_collection[key]):.4f} ± {np.std(metrics_collection[key]):.4f}")

    return models, metrics_collection

def evaluate_model(model, X_test, y_test, models=None, dataset_name="Test Set"):
    X_test_esm2, X_test_ankh = X_test
    model.eval()
    with torch.no_grad():
        X_test_esm2_tensor = torch.FloatTensor(X_test_esm2).unsqueeze(2).to(device)
        X_test_ankh_tensor = torch.FloatTensor(X_test_ankh).unsqueeze(2).to(device)
        print(f"{dataset_name} ESM2 tensor shape: {X_test_esm2_tensor.shape}")
        print(f"{dataset_name} Ankh tensor shape: {X_test_ankh_tensor.shape}")

        if models:
            for i, m in enumerate(models, 1):
                m.eval()
                outputs = m(X_test_esm2_tensor, X_test_ankh_tensor)
                predicted_probability = torch.sigmoid(outputs).cpu().numpy()
                predicted_class = (predicted_probability > 0.5).astype(int)
                metric, roc_data, prc_data = caculate_metric(predicted_class, y_test, predicted_probability)
                print(f"Model {i} - ACC: {metric[0]:.4f}, SP: {metric[3]:.4f}, SN: {metric[2]:.4f}, "
                      f"PRE: {metric[1]:.4f}, F1: {metric[4]:.4f}, MCC: {metric[6]:.4f}, "
                      f"AUC: {metric[5]:.4f}, AUPR: {prc_data[2]:.4f}")

        if models:
            weights = [1.0 / len(models)] * len(models)
            print(f"Ensemble weights: {weights}")
            probs = torch.zeros(len(y_test)).to(device)
            for m, w in zip(models, weights):
                m.eval()
                outputs = m(X_test_esm2_tensor, X_test_ankh_tensor)
                probs += w * torch.sigmoid(outputs)
            predicted_probability = probs.cpu().numpy()
            print(f"Using ensemble of {len(models)} models for prediction on {dataset_name}")
        else:
            outputs = model(X_test_esm2_tensor, X_test_ankh_tensor)
            predicted_probability = torch.sigmoid(outputs).cpu().numpy()
            print(f"Using single model for prediction on {dataset_name}")

        predicted_class = (predicted_probability > 0.5).astype(int)
        metric, roc_data, prc_data = caculate_metric(predicted_class, y_test, predicted_probability)
        metrics = {
            'ACC': metric[0],
            'PRE': metric[1],
            'SN': metric[2],
            'SP': metric[3],
            'F1': metric[4],
            'AUC': metric[5],
            'MCC': metric[6],
            'AUPR': prc_data[2],
            'fpr': roc_data[0],
            'tpr': roc_data[1],
            'precision': prc_data[1],
            'recall': prc_data[0]
        }

        print(f"\n{dataset_name} Results (Threshold=0.5):")
        for key in ['ACC', 'SP', 'SN', 'PRE', 'F1', 'MCC', 'AUC', 'AUPR']:
            print(f"{key}: {metrics[key]:.4f}")
        print(f"{dataset_name} Confusion Matrix:")
        print(confusion_matrix(y_test, predicted_class))
             
        # 保存 ROC 和 PR 曲线数据到 CSV 文件
        roc_curve_data = pd.DataFrame({
            'FPR': metrics['fpr'],
            'TPR': metrics['tpr']
        })
        pr_curve_data = pd.DataFrame({
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        })
        roc_curve_data.to_csv(f'./save/Test_Curve/roc_data_ankh-esm2-net-kan.csv', index=False)
        pr_curve_data.to_csv(f'./save/Test_Curve/pr_data_ankh-esm2-net-kan.csv', index=False)
        print(f"已保存 {dataset_name} 的 ROC 和 PR 曲线数据到 ./save/Test_Curve/")     
        
        # 分析误分类样本
        misclassified_idx = np.where(predicted_class != y_test)[0]
        if len(misclassified_idx) > 0:
            print(f"\nMisclassified samples (indices): {misclassified_idx}")
            print(f"True labels: {y_test[misclassified_idx]}")
            print(f"Predicted probabilities: {predicted_probability[misclassified_idx]}")

        return metrics

def plot_tsne_visualization(X_train, X_test, y_train, y_test, model, dataset_name="Train Set"):
    """
    绘制ESM2+Ankh原始特征和模型处理后特征的t-SNE可视化图（2行3列布局）
    """
    # 解析输入数据（使用采样后的训练集）
    X_data, y_data = X_train, y_train
    X_esm2, X_ankh = X_data
    y_data = np.array(y_data)  

    # 转换为张量并获取模型输出
    X_esm2_tensor = torch.FloatTensor(X_esm2).unsqueeze(2).to(device)
    X_ankh_tensor = torch.FloatTensor(X_ankh).unsqueeze(2).to(device)

    model.eval()
    with torch.no_grad():
        _, esm2_cnn_gru_out, ankh_cnn_gru_out, _, kan1_out = model(X_esm2_tensor, X_ankh_tensor, return_intermediate=True)
        esm2_cnn_gru_out = esm2_cnn_gru_out.cpu().numpy()
        ankh_cnn_gru_out = ankh_cnn_gru_out.cpu().numpy()
        kan1_out = kan1_out.cpu().numpy()

    # 准备t-SNE输入数据
    features = [
        ("Ankh Embedding", X_ankh),
        ("ESM-2 Embedding", X_esm2),
        ("Ankh+ESM-2 Embedding", np.concatenate([X_ankh, X_esm2], axis=1)),
        ("Ankh After CNN-BiGRU", ankh_cnn_gru_out),
        ("ESM-2 After CNN-BiGRU", esm2_cnn_gru_out),
        ("Concat After KAN", kan1_out)
    ]
    
    # 准备子图标题
    subplot_titles = [
        "Ankh Embedding",
        "ESM-2 Embedding",
        "Ankh+ESM-2 Embedding",
        "Ankh After CNN-BiGRU",
        "ESM-2 After CNN-BiGRU",
        "Concat After KAN"
    ]
    figure_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]  # 左上方序号标注

    # 设置t-SNE参数
    tsne = TSNE(n_components=2, random_state=seed_n, perplexity=30, n_iter=2000)

    # 创建2行3列的大图
    plt.figure(figsize=(21, 14), dpi=600)
    plt.rcParams.update({'xtick.labelsize': 14, 'ytick.labelsize': 14, 'font.family': 'DejaVu Sans'})
    
    # 分别计算并绘制每个特征的t-SNE
    for i, (title, data) in enumerate(features):
        # 计算t-SNE
        tsne_result = tsne.fit_transform(data)
        
        # 在大图中添加子图
        ax = plt.subplot(2, 3, i+1)
        
        # 绘制散点图
        scatter1 = ax.scatter(tsne_result[y_data == 1, 0], tsne_result[y_data == 1, 1], 
                  c='#1f77b4', marker='o', label='Positive', s=7, alpha=0.7)
        scatter2 = ax.scatter(tsne_result[y_data == 0, 0], tsne_result[y_data == 0, 1], 
                  c='#ff7f0e', marker='o', label='Negative', s=7, alpha=0.7)
        
        # 添加序号标注
        ax.text(0.01, 1.09, s=figure_labels[i], transform=ax.transAxes, 
                 ha='left', va='top', fontsize=24, fontweight='normal')  
        
        # 设置坐标轴标签和范围
        ax.set_xlabel("Dimension 1", fontsize=16)
        ax.set_ylabel("Dimension 2", fontsize=16)
        
        # 设置子图标题
        ax.text(0.5, 1.07, s=subplot_titles[i], transform=ax.transAxes, ha='center', va='top', fontsize=19)
        
        # 添加图例（调整位置避免遮挡）
        ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.9)
    
    # 调整整体布局
    plt.tight_layout()  
    
    # 保存大图
    file_name = f'tsne_all_features_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(f'./save/plots/{file_name}', format='png', dpi=600, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    print(f"所有特征的t-SNE可视化图已合并保存至 ./save/plots/{file_name}")

def main():
    print("加载数据...")
    X_train, X_test, y_train, y_test, scaler_esm2, scaler_ankh, pca_esm2, pca_ankh = get_train_test_data()

    X_train_esm2, X_train_ankh = X_train
    X_test_esm2, X_test_ankh = X_test
    print(f"训练数据形状: ESM2={X_train_esm2.shape}, Ankh={X_train_ankh.shape}")
    print(f"测试数据形状: ESM2={X_test_esm2.shape}, Ankh={X_test_ankh.shape}")
    print(f"训练集: {sum(y_train == 1)} 正样本, {sum(y_train == 0)} 负样本")
    print(f"测试集: {sum(y_test == 1)} 正样本, {sum(y_test == 0)} 负样本")
   
    all_folds_exist = all(os.path.exists(f'./save/model/fold_{i + 1}_ankh-esm2-net-kan.pt') for i in range(5))
    if all_folds_exist:
        print("\n加载五折模型进行集成预测...")
        models = []
        for i in range(5):
            fold_model = Net().to(device)
            fold_model.load_state_dict(torch.load(f'./save/model/fold_{i + 1}_ankh-esm2-net-kan.pt'))
            models.append(fold_model)
        model = models[0]
        print("已加载五折模型")
    else:
        print("\n开始五折交叉验证以训练和保存模型...")
        models, cv_results = train_five_fold_CV(X_train, y_train)
        model = models[0]

    print("\n在测试集上进行评估...")
    test_results = evaluate_model(model, X_test, y_test, models, dataset_name="Test Set")

    print("\n生成训练集的 t-SNE 可视化图...")
    plot_tsne_visualization(X_train, X_test, y_train, y_test, model, dataset_name="Train Set")

if __name__ == "__main__":
    main()