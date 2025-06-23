from torch import nn
from torch.nn import Sequential, Conv3d, InstanceNorm3d, LeakyReLU, Linear, Dropout, ReLU, Sigmoid
import torch.nn.functional as F
import torch


def masked_average_pooling(x, mask):
    # mask = (y == 1).to(torch.float)
    # mask = y

    eps = 1e-12

    if mask.size()[2:] != x.size()[2:]:
        mask = torch.nn.functional.interpolate(mask, x.size()[2:], mode='trilinear', align_corners=False)

    area = torch.sum(mask.view(mask.size(0), -1), dim=1, keepdim=True) + eps

    masked = x * mask
    masked = torch.sum(masked.view(masked.size(0), masked.size(1), -1), axis=2)
    masked = masked / area
    return masked


class ConvBlock(nn.Module):
    def __init__(self, in_channels=4, out_channels=5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(ConvBlock, self).__init__()
        self.conv = Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = InstanceNorm3d(out_channels, affine=True)
        self.relu = LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvBlocks(nn.Module):
    def __init__(self, in_channels=4, out_channels=5, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                 num_blocks=2):
        super(ConvBlocks, self).__init__()
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride, padding))
        for i in range(num_blocks - 1):
            layers.append(ConvBlock(out_channels, out_channels, kernel_size, padding=padding))

        self.main = Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        return x


class BaseNet(nn.Module):
    def __init__(self, nOut=5):
        super(BaseNet, self).__init__()
        nCh = 8
        self.conv = ConvBlocks(1, nCh)

        nFC = nCh * 4
        # self.conv_ = Linear(nFC, 4)

        self.fc1 = Linear(nCh, nFC)
        self.fc2 = Linear(nFC, nFC)
        self.fc3 = Linear(nFC, nOut)
        self.dropout = Dropout(inplace=True)
        self.sigmoid = Sigmoid()

    def forward(self, x, y):

        x = self.conv(x)
        x = masked_average_pooling(x, y)

        # x = x.view(1, -1)
        # att = self.conv_(x)
        # att = self.sigmoid(att)
        # att = att.unsqueeze(2).repeat(1, 1, 8).view(1, -1)
        # x = att * x


        x = x.view(1, -1)
        # x = self.dropout(F.relu(self.fc1(x)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x


class BaseNet_multi(nn.Module):
    def __init__(self):
        super(BaseNet_multi, self).__init__()
        nCh = 8
        self.conv = ConvBlocks(1, nCh)
        self.conv_2 = ConvBlocks(nCh, nCh*2, stride=2)

        nFC = nCh * 4
        nFC_2 = nCh * 8
        self.fc1 = Linear(nFC, nFC)
        self.fc2 = Linear(nFC, nFC)
        self.fc3 = Linear(nFC, 5)

        self.fc1_2 = Linear(nFC_2, nFC_2)
        self.fc2_2 = Linear(nFC_2, nFC_2)
        self.fc3_2 = Linear(nFC_2, 5)

        self.dropout = Dropout(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.relu = ReLU()

        # self.fc_final = Linear(nFC+nFC_2, 5)

    def forward(self, x, y):
        conv_q = self.conv(x)
        v = masked_average_pooling(conv_q, y)
        v = v.view(1, -1)
        conv_q_2 = self.conv_2(conv_q)
        v_2 = masked_average_pooling(conv_q_2, y)
        v_2 = v_2.view(1, -1)

        # out = self.dropout(self.relu(self.fc1(v)))
        # out = self.dropout(self.relu(self.fc2(out)))
        x = F.relu(self.fc1(v))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x_2 = F.relu(self.fc1_2(v_2))
        x_2 = F.relu(self.fc2_2(x_2))
        x_2 = self.fc3_2(x_2)

        # fused = torch.cat([x_1_feat, x_2_feat], dim=1)
        # x_final = self.fc_final(fused)

        return x, x_2 # , x_final


class TransNet(nn.Module):
    def __init__(self):
        super(TransNet, self).__init__()
        nCh = 8
        self.conv_q = ConvBlocks(1, nCh)
        self.conv_k = ConvBlocks(1, nCh)
        self.conv_v = ConvBlocks(1, nCh)

        nFC = nCh * 4
        self.fc1 = Linear(nFC, nFC)
        self.fc2 = Linear(nFC, nFC)
        self.fc3 = Linear(nFC, 5)

        # self.pe = nn.Parameter(torch.randn(4, nCh) * (nCh**-0.5))

        # self.dropout = Dropout(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.relu = ReLU()

    def forward(self, x, y):
        conv_q = self.conv_q(x)
        conv_k = self.conv_k(x)
        conv_v = self.conv_v(x)

        # mask = F.one_hot(y.type(torch.int64))
        # y = mask[:, :, :, :, :, 1] + mask[:, :, :, :, :, 2]

        q = masked_average_pooling(conv_q, y)
        k = masked_average_pooling(conv_k, y)
        v = masked_average_pooling(conv_v, y)

        # q = q + self.pe
        # k = k + self.pe
        # v = v + self.pe

        k_t = torch.transpose(k, 0, 1)
        scores = torch.matmul(q, k_t) * (8 ** -0.5)
        attn = self.softmax(scores)
        v = v + 0.1 * torch.matmul(attn, v)

        v = v.view(1, -1)

        # x = self.dropout(self.relu(self.fc1(v)))
        # x = self.dropout(self.relu(self.fc2(x)))
        x = F.relu(self.fc1(v))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TransNet_multi(nn.Module):
    def __init__(self):
        super(TransNet_multi, self).__init__()
        nCh = 8
        self.conv_q = ConvBlocks(1, nCh)
        self.conv_k = ConvBlocks(1, nCh)
        self.conv_v = ConvBlocks(1, nCh)

        self.conv_q_2 = ConvBlocks(nCh, nCh*2, stride=2)
        self.conv_k_2 = ConvBlocks(nCh, nCh*2, stride=2)
        self.conv_v_2 = ConvBlocks(nCh, nCh*2, stride=2)

        nFC = nCh * 4
        nFC_2 = nCh * 8
        self.fc1 = Linear(nFC, nFC)
        self.fc2 = Linear(nFC, nFC)
        self.fc3 = Linear(nFC, 5)

        self.fc1_2 = Linear(nFC_2, nFC_2)
        self.fc2_2 = Linear(nFC_2, nFC_2)
        self.fc3_2 = Linear(nFC_2, 5)

        self.pe = nn.Parameter(torch.randn(4, nCh) * (nCh**-0.5))
        self.pe_2 = nn.Parameter(torch.randn(4, nCh*2) * ((nCh*2) ** -0.5))

        self.dropout = Dropout(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.relu = ReLU()

        # self.fc_final = Linear(nFC+nFC_2, 5)

    def forward(self, x, y):
        conv_q = self.conv_q(x)
        conv_k = self.conv_k(x)
        conv_v = self.conv_v(x)

        # mask = F.one_hot(y.type(torch.int64))
        # y = mask[:, :, :, :, :, 1] + mask[:, :, :, :, :, 2]

        q = masked_average_pooling(conv_q, y)
        k = masked_average_pooling(conv_k, y)
        v = masked_average_pooling(conv_v, y)

        q = q + self.pe
        k = k + self.pe
        v = v + self.pe

        k_t = torch.transpose(k, 0, 1)
        scores = torch.matmul(q, k_t) * (8 ** -0.5)
        attn = self.softmax(scores)
        v = v + 0.1 * torch.matmul(attn, v)

        v = v.view(1, -1)

        conv_q_2 = self.conv_q_2(conv_q)
        conv_k_2 = self.conv_k_2(conv_k)
        conv_v_2 = self.conv_v_2(conv_v)

        q_2 = masked_average_pooling(conv_q_2, y)
        k_2 = masked_average_pooling(conv_k_2, y)
        v_2 = masked_average_pooling(conv_v_2, y)

        q_2 = q_2 + self.pe_2
        k_2 = k_2 + self.pe_2
        v_2 = v_2 + self.pe_2

        k_t_2 = torch.transpose(k_2, 0, 1)
        scores_2 = torch.matmul(q_2, k_t_2) * (16**-0.5)
        attn_2 = self.softmax(scores_2)
        v_2 = 0.1 * torch.matmul(attn_2, v_2) + v_2

        v_2 = v_2.view(1, -1)

        # out = self.dropout(self.relu(self.fc1(v)))
        # out = self.dropout(self.relu(self.fc2(out)))
        x = F.relu(self.fc1(v))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x_2 = F.relu(self.fc1_2(v_2))
        x_2 = F.relu(self.fc2_2(x_2))
        x_2 = self.fc3_2(x_2)

        # fused = torch.cat([x_1_feat, x_2_feat], dim=1)
        # x_final = self.fc_final(fused)

        return x, x_2, attn, attn_2 # , x_final
