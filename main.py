import os
import random

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from nilearn.connectome import ConnectivityMeasure
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cosine_similarity
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import rotate, shift
from sklearn.utils.class_weight import compute_class_weight
import argparse
from scipy.spatial import distance
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold  # 新增引用

import tool_data
import tool_graph
from model import CHW_GCN


def load_nifti_file(filepath):
    img = nib.load(filepath)
    return img.get_fdata()


def random_rotation(volume):
    return rotate(volume, angle=np.random.uniform(-10, 10), axes=(1, 2), reshape=False)


def random_shift(volume):
    return shift(volume, shift=np.random.uniform(-5, 5, size=3))


class DualModeProcessor:
    def __init__(self, args):
        self.args = args
        self.conn_measure = ConnectivityMeasure(kind='correlation')
        # self.mask = (np.triu_indices(110)[0], np.triu_indices(110)[1] + 1)
        self.mask = (np.triu_indices(110)[0], np.triu_indices(110)[1] + 1)

    def _process_batch(self, time_batch, mode='SA'):
        """批处理时间序列"""
        batch_features = []
        stride = 15
        t = 15  # 每个被试生成15个时间窗口

        for ts in time_batch:
            if mode == 'SA':
                # SA模式处理（还原Pre_time_SA.py逻辑）
                windowsize = 30
                for i in range(t):
                    k = stride * i
                    # 处理窗口越界
                    if k + windowsize >= ts.shape[0]:
                        k = random.randint(1, ts.shape[0] - windowsize - 2)

                    # 提取窗口并计算功能连接
                    window = ts[k:k + windowsize]
                    fc = self.conn_measure.fit_transform([window])[0]
                    batch_features.append(fc[self.mask])

            else:
                # MA模式处理（还原Pre_time_MA.py逻辑）
                windowsizes = [10, 20, 30, 40, 50]
                window_point_Time = []

                # 为每个窗口大小生成时间点
                for size in windowsizes:
                    points = []
                    for i in range(t):
                        base_k = size // 2 + stride * i  # 基准中心点
                        # 调整窗口位置防止越界
                        if base_k + size // 2 >= ts.shape[0]:
                            base_k = random.randint(size // 2, ts.shape[0] - size // 2 - 1)
                        begin = base_k - size // 2
                        end = base_k + size // 2
                        points.append((begin, end))
                    window_point_Time.append(points)

                # 生成所有窗口特征
                for size_idx, size in enumerate(windowsizes):
                    for window_idx in range(t):
                        begin, end = window_point_Time[size_idx][window_idx]
                        window = ts[begin:end]
                        fc = self.conn_measure.fit_transform([window])[0]
                        batch_features.append(fc[self.mask])

        # 重构为原始数据结构 [时间窗口数 × 被试数 × 特征维度]
        features_array = np.array(batch_features)

        # 调整维度顺序
        if mode == 'SA':
            # SA模式: [被试数 × 时间窗口数 × 特征] → [时间窗口数 × 被试数 × 特征]
            return features_array.reshape(len(time_batch), t, -1).transpose(1, 0, 2)
        else:
            # MA模式: [总窗口数 × 被试数 × 特征]
            return features_array.reshape(len(windowsizes) * t, len(time_batch), -1)

    def _build_graphs(self, features):
        """构建图结构（完整还原原始PCA流程）"""
        # 拼接所有特征用于PCA拟合
        all_features = np.concatenate(features, axis=0)

        # PCA降维（与原始脚本完全一致）
        pca = PCA(n_components=self.args.input_dim)
        pca.fit(all_features)

        # 处理每个被试的特征
        graphs = []
        for i, feat in enumerate(features):
            # 应用PCA转换
            feat_pca = pca.transform(feat)

            # 计算相关距离矩阵
            dist = distance.squareform(
                distance.pdist(feat_pca, metric='correlation')
            )

            # 计算sigma并构建基础图
            sigma = np.nanmean(dist)
            graph = np.exp(-dist ** 2 / (2 * sigma ** 2))

            # 结合先验图特征
            # final_graph = graph * self.base_graph
            final_graph = graph
            # final_graph = sparse.coo_matrix(final_graph.astype(np.float32))

            # # 添加自环并标准化
            # final_graph += np.eye(final_graph.shape[0])  # 添加自环
            # row_sum = final_graph.sum(axis=1, keepdims=True)
            # final_graph = final_graph / np.sqrt(row_sum * row_sum.T)  # 对称标准化

            graphs.append(final_graph)

        return np.array(graphs)


class BrainDataset(Dataset):
    def __init__(self, data_dir, data, args, augment=False):
        """
        修改：data 参数现在既可以接收 csv 路径(str)，也可以接收 DataFrame 对象
        """
        self.args = args
        self.data_dir = data_dir

        # 判断传入的是路径还是DataFrame
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.reset_index(drop=True)  # 重置索引以防止iloc切片后的索引不连续问题

        self.augment = augment
        self.subjects = self.df['SUB_ID'].values
        self.labels = (self.df['DX_GROUP'] - 1).values  # ABIDE&PD
        # self.labels = (self.df['DX_GROUP']).values # ADHD

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, f"{self.subjects[idx]}.nii.gz")
        # volume = load_nifti_file(file_path)
        # volume = (volume - np.mean(volume)) / np.std(volume)
        #
        # if self.augment:
        #     if np.random.rand() > 0.5:
        #         volume = random_rotation(volume)
        #     if np.random.rand() > 0.5:
        #         volume = random_shift(volume)
        #
        # volume = torch.FloatTensor(volume).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor([self.labels[idx]])
        Time_SA = []
        Time_MA = []
        path = os.path.join(self.args.data_path, f"{self.subjects[idx]}.1D")
        # path = os.path.join(self.args.data_path, f"{self.subjects[idx]}.npy")
        feature = []
        if os.path.exists(path):
            i = 0
            # ts = np.load(path)
            # Time_SA.append(ts)
            # Time_MA.append(ts)
            with open(path, "r") as file:
                for line in file:
                    if i == 0:
                        i += 1
                        continue
                    temp = line.strip().split('\t')  # 116
                    feature.append([float(x) for x in temp])
            Time_SA.append(np.array(feature))
            Time_MA.append(np.array(feature))
        # if os.path.exists(path):
        #     loadData = np.load(path)
        #     for line in loadData:
        #         line_list = list(line)
        #         feature.append([float(x) for x in line_list])
        #     Time_SA.append(np.array(feature))
        #     Time_MA.append(np.array(feature))
        process = DualModeProcessor(self.args)
        sa_features = process._process_batch(Time_SA, 'SA')
        ma_features = process._process_batch(Time_MA, 'MA')
        return label, sa_features, ma_features


class DualGraph:
    def __init__(self, args):
        self.args = args

    def sa_gragh(self, graphs):
        graph_list_list = []
        graph_gl_list_list = []

        for b in range(1, self.args.knn + 1):  # 节点之间的关系
            graph_list = [tool_graph.nor_graph(graphs[k], topk=b, w=0.5) for k in range(self.args.nb_t)]
            graph_gl_list = [tool_graph.torch2dgl(A) for A in graph_list]
            graph_list_list.append(graph_list)
            graph_gl_list_list.append(graph_gl_list)

        return graph_gl_list_list

    def ma_gragh(self, graphs):
        graph_list_list_N = []
        graph_gl_list_list_N = []
        for b in range(1, self.args.knn + 1):
            graph_list_N = [tool_graph.nor_graph(graphs[k], topk=b, w=0.5) for k in range(self.args.nb_t_N * 5)]
            graph_gl_list_N = [tool_graph.torch2dgl(A) for A in graph_list_N]
            graph_list_list_N.append(graph_list_N)
            graph_gl_list_list_N.append(graph_gl_list_N)

        return graph_gl_list_list_N

    def build_data(self, sa_features, sa_graphs, ma_features, ma_graphs):
        sa_feature_permuted = sa_features.permute(1, 0, 2)
        ma_feature_permuted = ma_features.permute(1, 0, 2)
        sa_graphs = sa_graphs.permute(1, 0, 2)
        ma_graphs = ma_graphs.permute(1, 0, 2)
        return sa_feature_permuted, sa_graphs, ma_feature_permuted, ma_graphs


def plot_history(train_loss, val_loss, train_acc, val_acc, fold_idx):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title(f'Model Accuracy (Fold {fold_idx})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title(f'Model Loss (Fold {fold_idx})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    # 修改保存路径，带上 Fold 编号
    plt.savefig(f'../result2/training_metrics_GBConv_fold{fold_idx}.png')
    plt.close()  # 关闭图片防止内存泄漏
    # plt.show()


def main():
    data_dir = r'F:\ABIDE_fmri\3D_data'
    csv_path_all = '../data/ABIDE_csv/all_data.csv'


    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--res_dir', type=str, default='result2/')
    parser.add_argument('--dataset', type=str, default='Ori_ABIDE',
                        choices=['Ori_ABIDE', 'Ori_PD'])
    parser.add_argument('--save_dir', type=str, default='modelsave/')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--label_rate', type=float, default=0.2)

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--data_path', type=str, default=r'E:\ASD',
                        help='Path to raw fMRI data')
    parser.add_argument('--window_size_SA', type=int, default=30,
                        help='Window size for SA mode')
    parser.add_argument('--window_size_MA', nargs='+', type=int,
                        default=[10, 20, 30, 40, 50],
                        help='Window sizes for MA mode')
    parser.add_argument('--nb_t_N', type=int, default=15,
                        help='')
    parser.add_argument('--nb_t', type=int, default=15,
                        help='')
    parser.add_argument('--input_dim', type=int, default=256)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--beta', type=int, default=0.2)
    parser.add_argument('--lastdim', type=int, default=256)
    parser.add_argument('--plus', type=int, default=1)
    parser.add_argument('--lastdim2', type=int, default=512)
    parser.add_argument('--epoch_all1', type=int, default=1)
    parser.add_argument('--epoch_all2', type=int, default=1)
    parser.add_argument('--knn', type=int, default=5)
    parser.add_argument('--lr1', type=float, default=0.001)  # 这里应该是 float
    parser.add_argument('--lr2', type=float, default=0.0001)
    parser.add_argument('--weight_decay1', type=float, default=1e-5)
    parser.add_argument('--weight_decay2', type=float, default=1e-5)
    parser.add_argument('--random_aug_feature', type=int, default=3)
    parser.add_argument('--random_aug_edge', type=int, default=3)
    parser.add_argument('--SAMA', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=30,
                        help='')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 确保结果文件夹存在
    if not os.path.exists('../result2'):
        os.makedirs('../result2')

    # 读取全量数据（如果使用上面注释的合并代码，这里直接用 full_df）
    if 'full_df' not in locals():
        full_df = pd.read_csv(csv_path_all)

    labels = (full_df['DX_GROUP'] - 1).values  # 获取标签用于分层采样

    # 初始化 5折交叉验证
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    # 存储每一折的结果
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }

    # 开始交叉验证循环
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'=' * 20} Fold {fold + 1}/{n_folds} {'=' * 20}")

        # 根据索引切分数据
        train_df_fold = full_df.iloc[train_idx]
        val_df_fold = full_df.iloc[val_idx]

        # 构建Dataset
        train_dataset = BrainDataset(data_dir, train_df_fold, args, augment=True)
        val_dataset = BrainDataset(data_dir, val_df_fold, args)  # 验证集即为本折的测试集

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 模型初始化 (每一折都要重置模型和优化器)
        model = CHW_GCN(
            in_dim=3025,
            hid_dim=args.lastdim * args.plus,
            out_dim=args.lastdim,
            out_dim2=args.lastdim2,
            n_layers=1,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr1, weight_decay=args.weight_decay1)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        # 类别权重 (基于当前折的训练集计算)
        cur_train_labels = (train_df_fold['DX_GROUP'] - 1).values
        class_weights = compute_class_weight('balanced', classes=np.unique(cur_train_labels), y=cur_train_labels)
        class_weights = torch.FloatTensor(class_weights).to(device)

        # 训练循环
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        choose = [0, args.nb_t_N * 1, args.nb_t_N * 3, args.nb_t_N * 4]

        # 当前折的最佳指标记录
        best_fold_acc = 0.0
        best_fold_f1 = 0.0
        final_fold_metrics = {}  # 存储最后一次评估的指标

        for epoch in tqdm(range(args.epochs)):
            model.train()
            epoch_loss = 0.0
            correct = 0
            train_pred = []
            train_true = []

            for labels_batch, sa_features, ma_features in train_loader:

                labels_batch = labels_batch.squeeze().to(device)
                sa_features = sa_features.to(device)
                ma_features = ma_features.to(device)

                sa_feat = sa_features.squeeze(2).permute(1, 0, 2)
                ma_feat = ma_features.squeeze(2).permute(1, 0, 2)
                processor_1 = DualModeProcessor(args)
                sa_graphs = processor_1._build_graphs(sa_feat.cpu().numpy())
                ma_graphs = processor_1._build_graphs(ma_feat.cpu().numpy())
                processor_2 = DualGraph(args)
                sa_graph = processor_2.sa_gragh(torch.from_numpy(sa_graphs))
                ma_graph = processor_2.ma_gragh(torch.from_numpy(ma_graphs))

                feat1, graph1 = sa_feat, sa_graph
                feat2, graph2 = ma_feat, ma_graph

                k = random.randint(0, args.nb_t_N - 1)
                id_2_N = choose[random.randint(0, 3)] + k
                id_1 = random.randint(0, args.nb_t - 2)
                id_2 = id_1 + 1
                feat1_ori = feat1[id_1]
                feat2_ori = feat1[id_2]
                feat2_ori_N = feat2[id_2_N]
                graph2_ori_N = graph2[-1][id_2_N]
                graph1_ori = graph1[0][id_1]
                graph2_ori = graph1[-1][id_2]

                if args.SAMA:
                    feat2_ori = random.sample([feat2_ori, feat2_ori_N], 1)[0]
                    graph2_ori = random.sample([graph2_ori_N, graph2_ori], 1)[0]

                # 数据增强
                graph1, feat1 = tool_data.RA(graph1_ori.cpu(), feat1_ori, args.random_aug_feature / 10,
                                             args.random_aug_edge / 10)
                graph2, feat2 = tool_data.RA(graph2_ori.cpu(), feat2_ori, args.random_aug_feature / 10,
                                             args.random_aug_edge / 10)

                graph1 = graph1.add_self_loop()
                graph2 = graph2.add_self_loop()
                graph1 = graph1.to(device)
                graph2 = graph2.to(device)
                feat1 = feat1.to(device)
                feat2 = feat2.to(device)
                N = feat1.size(1)

                optimizer.zero_grad()

                with autocast():
                    embeding, embeding_a, embeding_b, p1, p2 = model(graph1, feat1, graph2, feat2)
                    c1 = torch.mm(embeding_a.T, embeding_a) / N
                    c2 = torch.mm(embeding_b.T, embeding_b) / N
                    loss1 = criterion(embeding, labels_batch)

                    I_target = torch.tensor(np.eye(c1.size()[0])).to(device)

                    loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
                    loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
                    loss = 1 - args.alpha * cosine_similarity(embeding_a, embeding_b.detach(),
                                                              dim=-1).mean() + args.beta * (
                                   loss_c1 + loss_c2) + loss1
                    # loss = loss * class_weights[labels].mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                _, preds = torch.max(embeding, 1)
                correct += (preds == labels_batch).sum().item()
                train_pred.extend(preds.cpu().tolist())
                train_true.extend(labels_batch.cpu().tolist())

            train_loss.append(epoch_loss / len(train_loader))
            train_acc.append(correct / len(train_dataset))

            # 验证 (在此Fold的验证集上进行)
            model.eval()
            val_epoch_loss = 0.0
            val_correct = 0
            val_pred = []
            val_true = []

            with torch.no_grad():
                for labels_batch, sa_features, ma_features in val_loader:

                    labels_batch = labels_batch.squeeze().to(device)
                    sa_features = sa_features.to(device)
                    ma_features = ma_features.to(device)

                    sa_feat = sa_features.squeeze(2).permute(1, 0, 2)
                    ma_feat = ma_features.squeeze(2).permute(1, 0, 2)
                    processor_1 = DualModeProcessor(args)
                    sa_graphs = processor_1._build_graphs(sa_feat.cpu().numpy())
                    ma_graphs = processor_1._build_graphs(ma_feat.cpu().numpy())
                    processor_2 = DualGraph(args)
                    sa_graph = processor_2.sa_gragh(torch.from_numpy(sa_graphs))
                    ma_graph = processor_2.ma_gragh(torch.from_numpy(ma_graphs))

                    feat1, graph1 = sa_feat, sa_graph
                    feat2, graph2 = ma_feat, ma_graph

                    k = random.randint(0, args.nb_t_N - 1)
                    id_2_N = choose[random.randint(0, 3)] + k
                    id_1 = random.randint(0, args.nb_t - 2)
                    id_2 = id_1 + 1
                    feat1_ori = feat1[id_1]
                    feat2_ori = feat1[id_2]
                    feat2_ori_N = feat2[id_2_N]
                    graph2_ori_N = graph2[-1][id_2_N]
                    graph1_ori = graph1[0][id_1]
                    graph2_ori = graph1[-1][id_2]

                    if args.SAMA:
                        feat2_ori = random.sample([feat2_ori, feat2_ori_N], 1)[0]
                        graph2_ori = random.sample([graph2_ori_N, graph2_ori], 1)[0]

                    # 数据增强
                    graph1, feat1 = tool_data.RA(graph1_ori.cpu(), feat1_ori, args.random_aug_feature / 10,
                                                 args.random_aug_edge / 10)
                    graph2, feat2 = tool_data.RA(graph2_ori.cpu(), feat2_ori, args.random_aug_feature / 10,
                                                 args.random_aug_edge / 10)

                    graph1 = graph1.add_self_loop()
                    graph2 = graph2.add_self_loop()
                    graph1 = graph1.to(device)
                    graph2 = graph2.to(device)
                    feat1 = feat1.to(device)
                    feat2 = feat2.to(device)
                    N = feat1.size(1)

                    embeding, embeding_a, embeding_b, p1, p2 = model(graph1, feat1, graph2, feat2)
                    loss1 = criterion(embeding, labels_batch)
                    c1 = torch.mm(embeding_a.T, embeding_a) / N
                    c2 = torch.mm(embeding_b.T, embeding_b) / N

                    I_target = torch.tensor(np.eye(c1.size()[0])).to(device)

                    loss_c1 = (I_target - c1).pow(2).mean() + torch.diag(c1).mean()
                    loss_c2 = (I_target - c2).pow(2).mean() + torch.diag(c2).mean()
                    loss = 1 - args.alpha * cosine_similarity(embeding_a, embeding_b.detach(),
                                                              dim=-1).mean() + args.beta * (
                                   loss_c1 + loss_c2) + loss1

                    val_epoch_loss += loss.item()
                    _, preds = torch.max(embeding, 1)
                    val_correct += (preds == labels_batch).sum().item()
                    val_pred.extend(preds.cpu().tolist())
                    val_true.extend(labels_batch.cpu().tolist())

            val_loss.append(val_epoch_loss / len(val_loader))
            val_acc.append(val_correct / len(val_dataset))

            print(f'Epoch {epoch + 1}/{args.epochs}')
            print(f'Train Loss: {train_loss[-1]:.4f} Acc: {train_acc[-1]:.4f}')
            # 简单打印
            # print(f'Val Loss: {val_loss[-1]:.4f} Acc: {val_acc[-1]:.4f}\n')

            # 计算当前epoch的验证集详细指标
            cur_acc = accuracy_score(val_true, val_pred)
            cur_prec = precision_score(val_true, val_pred, average='micro')
            cur_rec = recall_score(val_true, val_pred, average='micro')
            cur_f1 = f1_score(val_true, val_pred, average='micro')
            try:
                cur_auc = roc_auc_score(val_true, val_pred, average='micro')
            except:
                cur_auc = 0.0  # 处理可能出现的错误（例如只有一类）

            print(f"Val Acc: {cur_acc:.4f} | F1: {cur_f1:.4f} | AUC: {cur_auc:.4f}\n")

            # 记录最后一个epoch的指标作为该Fold的结果（或者你可以添加逻辑保存最佳val acc的指标）
            final_fold_metrics = {
                'accuracy': cur_acc,
                'precision': cur_prec,
                'recall': cur_rec,
                'f1': cur_f1,
                'auc': cur_auc,
                'true': val_true,
                'pred': val_pred
            }

        # --- End of Epoch Loop for this Fold ---

        # 保存该Fold的模型
        torch.save(model.state_dict(), f'3d_attention_resnet_model_fold{fold + 1}.pth')

        # 绘制该Fold的训练曲线
        plot_history(train_loss, val_loss, train_acc, val_acc, fold + 1)

        # 生成该Fold的混淆矩阵
        plt.figure()
        sns.heatmap(confusion_matrix(final_fold_metrics['true'], final_fold_metrics['pred']), annot=True, fmt='d')
        plt.savefig(f'../result2/confusion_matrix_GBConv_fold{fold + 1}.png')
        plt.close()

        # 记录汇总指标
        fold_results['accuracy'].append(final_fold_metrics['accuracy'])
        fold_results['precision'].append(final_fold_metrics['precision'])
        fold_results['recall'].append(final_fold_metrics['recall'])
        fold_results['f1'].append(final_fold_metrics['f1'])
        fold_results['auc'].append(final_fold_metrics['auc'])

    # --- End of CV Loop ---

    print("\n" + "=" * 50)
    print("5-Fold Cross Validation Results Summary")
    print("=" * 50)
    print(f"Mean Accuracy : {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")
    print(f"Mean Precision: {np.mean(fold_results['precision']):.4f} ± {np.std(fold_results['precision']):.4f}")
    print(f"Mean Recall   : {np.mean(fold_results['recall']):.4f} ± {np.std(fold_results['recall']):.4f}")
    print(f"Mean F1 Score : {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
    print(f"Mean AUC      : {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()