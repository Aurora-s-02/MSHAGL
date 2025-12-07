import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, EdgeConv, GATConv
import torch

class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.act_fn(x)
        x = self.layer2(x)

        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim))
            self.convs.append(GraphConv(hid_dim, hid_dim))

        self.layer1 = nn.Linear(hid_dim, out_dim, bias=True)

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.elu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)
        x = self.layer1(F.elu(x))
        return x

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GATConv(in_dim, hid_dim, num_heads = 3))


        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GATConv(hid_dim, hid_dim, num_heads = 3))
            self.convs.append(GATConv(hid_dim, hid_dim, num_heads = 3))

        self.layer1 = nn.Linear(hid_dim, out_dim, bias=True)

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.elu(torch.mean(self.convs[i](graph, x), dim=1))
        x = torch.mean(self.convs[-1](graph, x), dim=1)
        x = self.layer1(F.elu(x))
        return x

class GATE_GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim,  weight = 'true'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim))
            self.convs.append(GraphConv(hid_dim, out_dim))

        # self.layer0 = nn.Linear(in_dim, hid_dim)
        # self.bn0 = nn.BatchNorm1d(hid_dim, affine=False)
        self.layer1 = nn.Linear(out_dim, out_dim)

    def forward(self, graph, x):
        x = x.float()
        for i in range(self.n_layers - 1):
            x = F.elu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)
        x = self.layer1(F.elu(x))
        return x

class CHW_GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, out_dim2, n_layers):
        super().__init__()
        self.backbone_1 = GATE_GCN(3025, hid_dim, out_dim, n_layers)
        self.backbone_all = GATE_GCN(6105, hid_dim, out_dim, n_layers)
        self.cls = MLP_Predictor(out_dim, 2, out_dim2)
        # self.CNN3D = GBConvNet3D()
        self.gate = MultiHeadDynamicGatedFusion(out_dim)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        self.matrix_size = 110
        self.triu_size = (self.matrix_size * (self.matrix_size + 1)) // 2  # 6786

        self.left_indices = torch.tensor([i for i in range(0, self.matrix_size, 2)], dtype=torch.long)
        self.right_indices = torch.tensor([i for i in range(1, self.matrix_size, 2)], dtype=torch.long)
        self.hgu = HemisphericGatingUnit(feature_dim=256)

    def get_embedding(self, cnn_input, A_a, X_a, A_b, X_b):
        embeding_a = self.backbone_all(A_a, X_a)  # (64,256)
        embeding_b = self.backbone_all(A_b, X_b)

        out = self.gate(embeding_a, embeding_b)
        output = out.detach()
        return output

    def finetune(self, cnn_input, A_a, X_a, A_b, X_b):
        # X_a_c = self.reconstruct_symmetric_matrix_efficient(X_a)
        embeding_a = self.backbone_all(A_a, X_a)  # (64,256)
        embeding_b = self.backbone_all(A_b, X_b)

        out = self.gate(embeding_a, embeding_b)
        out = F.elu(out)
        c = self.cls(out)
        return c

    def reconstruct_symmetric_matrix_efficient(self, triu_flat):

        batch_size = triu_flat.size(0)
        # 确保输入尺寸正确
        assert triu_flat.size(1) == self.triu_size, \
            f"输入尺寸错误: 期望 {self.triu_size} 个元素, 实际得到 {triu_flat.size(1)}"

        full_matrix = torch.zeros(
            (batch_size, self.matrix_size, self.matrix_size),
            device=triu_flat.device,
            dtype=torch.float32
        )

        rows, cols = torch.triu_indices(
            self.matrix_size,
            self.matrix_size,
            offset=0
        )

        triu_flat = triu_flat.to(torch.float32)

        full_matrix[:, rows, cols] = triu_flat

        lower_rows, lower_cols = torch.tril_indices(
            self.matrix_size,
            self.matrix_size,
            offset=-1
        )
        full_matrix[:, lower_rows, lower_cols] = full_matrix[:, lower_cols, lower_rows]

        return full_matrix

    def split_brain_features(self, full_brain_matrix):

        left_indices = self.left_indices.to(full_brain_matrix.device)
        right_indices = self.right_indices.to(full_brain_matrix.device)

        # 提取左脑特征 (双数索引)
        left_brain = full_brain_matrix.index_select(1, left_indices)  # 选择行
        left_brain = left_brain.index_select(2, left_indices)  # 选择列

        # 提取右脑特征 (单数索引)
        right_brain = full_brain_matrix.index_select(1, right_indices)  # 选择行
        right_brain = right_brain.index_select(2, right_indices)  # 选择列

        # 展平特征
        batch_size = full_brain_matrix.size(0)
        left_features = left_brain.reshape(batch_size, -1)
        right_features = right_brain.reshape(batch_size, -1)

        return left_features, right_features

    def forward(self, A_a, X_a, A_b, X_b):
        X_a = F.dropout(X_a, 0.2)
        X_b = F.dropout(X_b, 0.2)
        embeding_a = self.backbone_all(A_a, X_a)
        embeding_b = self.backbone_all(A_b, X_b)

        embeding_a = (embeding_a - embeding_a.mean(0)) / embeding_a.std(0)
        embeding_b = (embeding_b - embeding_b.mean(0)) / embeding_b.std(0)

        embeding_all = self.gate(embeding_a, embeding_b)

        c_a = self.cls(embeding_a)
        c_b = self.cls(embeding_b)
        all = self.fc(embeding_all)
        return all, embeding_a, embeding_b, c_a, c_b

class MLP_Predictor(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()


        self.net = nn.Sequential(
            nn.Linear(input_size, output_size, bias=True)
        )

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class HemisphericGatingUnit(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=128):
        """
        动态脑特征融合门控单元
        :param feature_dim: 输入特征维度 (默认256)
        :param hidden_dim: 隐藏层维度 (默认128)
        """
        super().__init__()

        self.gate_network = nn.Sequential(
            nn.Linear(3 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * feature_dim),
        )

        self.enhancer = nn.Linear(feature_dim, feature_dim)

    def forward(self, whole_feat, left_feat, right_feat):
        """
        :param left_feat: 左脑特征 (batch_size, feature_dim)
        :param right_feat: 右脑特征 (batch_size, feature_dim)
        :param whole_feat: 全脑特征 (batch_size, feature_dim)
        :return: 融合后的全脑增强特征 (batch_size, feature_dim)
        """
        batch_size = left_feat.size(0)

        # 1. 特征拼接 (沿特征维度)
        combined = torch.cat([left_feat, right_feat, whole_feat], dim=1)  # (128, 768)

        # 2. 生成特征级门控信号
        gate_scores = self.gate_network(combined)  # (128, 768)
        gate_scores = gate_scores.view(batch_size, 3, -1)  # (128, 3, 256)

        # 3. 通道维度Softmax归一化
        gate_weights = F.softmax(gate_scores, dim=1)  # (128, 3, 256)

        # 4. 特征加权融合
        weighted_left = gate_weights[:, 0, :] * left_feat
        weighted_right = gate_weights[:, 1, :] * right_feat
        weighted_whole = gate_weights[:, 2, :] * whole_feat
        fused_feat = weighted_left + weighted_right + weighted_whole  # (128, 256)


        enhanced_whole = self.enhancer(whole_feat)  # (128, 256)
        output = fused_feat + enhanced_whole  # (128, 256)

        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(concat))
        return x * y


class AttentionResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

        # 调整shortcut匹配维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.ca(x)
        x = self.sa(x)
        x += residual
        return F.relu(x)


class AttentionResNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        # 逐步增加通道数并下采样
        self.block1 = AttentionResBlock(32, 64, stride=2)
        self.block2 = AttentionResBlock(64, 128, stride=2)
        self.block3 = AttentionResBlock(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return x
        # return self.fc(x)

class MultiHeadDynamicGatedFusion(nn.Module): # 多头门控
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        self.head_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, 2),
                nn.Softmax(dim=1)
            ) for _ in range(num_heads)
        ])
        self.fusion_layer = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, emb_a, emb_b):
        combined = torch.cat([emb_a, emb_b], dim=1)
        all_heads = []
        for head in self.head_proj:
            gates = head(combined)
            weighted = torch.stack([emb_a, emb_b], 1) * gates.unsqueeze(2)
            all_heads.append(torch.sum(weighted, 1))

        return self.fusion_layer(torch.cat(all_heads, dim=1))


class ElementwiseDynamicGatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 3),
            nn.ReLU(),
            nn.Linear(embed_dim * 3, embed_dim * 3),
            nn.Sigmoid()
        )

    def forward(self, emb_a, emb_b, emb_cnn):
        combined = torch.cat([emb_a, emb_b, emb_cnn], dim=1)
        gates = self.gate_network(combined)

        gate_a, gate_b, gate_cnn = gates.chunk(3, dim=1)

        return emb_a * gate_a + emb_b * gate_b + emb_cnn * gate_cnn


class DeepDynamicGatedFusion(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        # 深层门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, emb_a, emb_b, emb_cnn):
        combined = torch.cat([emb_a, emb_b, emb_cnn], dim=1)
        gates = self.gate_network(combined)
        weighted_embs = torch.stack([emb_a, emb_b, emb_cnn], dim=1) * gates.unsqueeze(2)
        return torch.sum(weighted_embs, dim=1)

class BottConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.pointwise_1 = nn.Conv3d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv3d(mid_channels, mid_channels, kernel_size, stride, padding,
                                 groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv3d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x

def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)


class GBConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_type='GN'):
        super().__init__()
        self.stride = stride


        mid_channels = out_channels // 8
        self.block1 = nn.Sequential(
            BottConv3D(in_channels, out_channels, mid_channels, kernel_size=3,
                       stride=stride, padding=1),  # 使用stride下采样
            get_norm_layer(norm_type, out_channels, out_channels // 16),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            BottConv3D(out_channels, out_channels, mid_channels, kernel_size=3, padding=1),
            get_norm_layer(norm_type, out_channels, out_channels // 16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            BottConv3D(in_channels, out_channels, mid_channels, kernel_size=1,
                       stride=stride, padding=0),
            get_norm_layer(norm_type, out_channels, out_channels // 16),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            BottConv3D(out_channels, out_channels, mid_channels, kernel_size=1, padding=0),
            get_norm_layer(norm_type, out_channels, 16),
            nn.ReLU()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)

        return x + residual


class GBConvNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.gbconv_layers = nn.Sequential(
            GBConv3D(32, 64, stride=2),  # 32→64
            GBConv3D(64, 128, stride=2),  # 64→128
            GBConv3D(128, 256, stride=2)  # 128→256
        )

        self.avgpool = nn.AdaptiveAvgPool3d(1)


    def forward(self, x):
        x = self.initial(x)
        x = self.gbconv_layers(x)
        x = self.avgpool(x).view(x.size(0), -1)
        # return self.fc(x)
        return x