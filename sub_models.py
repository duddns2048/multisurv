"""MultiSurv sub-models."""

from bisect import bisect_left

import torch
import torch.nn as nn
from torchvision import models

from embrace_net import EmbraceNet
from attention import Attention

# model.py
from torch.nn import Sequential, Conv3d, InstanceNorm3d, LeakyReLU, Linear, Dropout, ReLU, Sigmoid
import torch.nn.functional as F




def freeze_layers(model, up_to_layer=None):
    if up_to_layer is not None:
        # Freeze all layers
        for i, param in model.named_parameters():
            param.requires_grad = False

        # Release all layers after chosen layer
        frozen_layers = []
        for name, child in model.named_children():
            if up_to_layer in frozen_layers:
                for params in child.parameters():
                    params.requires_grad = True
            else:
                frozen_layers.append(name)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        freeze_layers(self.model, up_to_layer='layer3')
        self.n_features = self.model.fc.in_features

        # Remove classifier (last layer)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x):
        out = self.model(x)

        return out


class FC(nn.Module):
    "Fully-connected model to generate final output."
    def __init__(self, in_features, out_features, n_layers, dropout=False,
                 batchnorm=True, scaling_factor=4):
        super(FC, self).__init__()
        if n_layers == 1:
            layers = self._make_layer(in_features, out_features, dropout,
                                      batchnorm)
        elif n_layers > 1:
            n_neurons = self._pick_n_neurons(in_features)
            if n_neurons < out_features:
                n_neurons = out_features

            if n_layers == 2:
                layers = self._make_layer(
                    in_features, n_neurons, dropout, batchnorm=batchnorm)
                layers += self._make_layer(
                    n_neurons, out_features, dropout, batchnorm)
            else:
                for layer in range(n_layers):
                    last_layer_i = range(n_layers)[-1]

                    if layer == 0:
                        n_neurons *= scaling_factor
                        layers = self._make_layer(
                            in_features, n_neurons, dropout, batchnorm=batchnorm)
                    elif layer < last_layer_i:
                        n_in = n_neurons
                        n_neurons = self._pick_n_neurons(n_in)
                        if n_neurons < out_features:
                            n_neurons = out_features
                        layers += self._make_layer(
                            n_in, n_neurons, dropout, batchnorm=batchnorm)
                    else:
                        layers += self._make_layer(
                            n_neurons, out_features, dropout, batchnorm)
        else:
            raise ValueError('"n_layers" must be positive.')

        self.fc = nn.Sequential(*layers)

    def _make_layer(self, in_features, out_features, dropout, batchnorm):
        layer = nn.ModuleList()
        if dropout:
            layer.append(nn.Dropout())
        layer.append(nn.Linear(in_features, out_features))
        layer.append(nn.ReLU(inplace=True))
        if batchnorm:
            layer.append(nn.BatchNorm1d(out_features))

        return layer

    def _pick_n_neurons(self, n_features):
        # Pick number of features from list immediately below n input
        n_neurons = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        idx = bisect_left(n_neurons, n_features)

        return n_neurons[0 if idx == 0 else idx - 1]

    def forward(self, x):
        return self.fc(x)

    
class ClinicalNet(nn.Module):
    """Clinical data extractor.

    Handle continuous features and categorical feature embeddings.
    """
    def __init__(self, output_vector_size, 
                #  embedding_dims=[(33, 17), (2, 1), (8, 4), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2), (20, 10)]
                #  embedding_dims=[(2, 1), (5, 3), (2, 1), (2, 1), (3, 2)]):
                 embedding_dims=[(2, 1), (5, 3), (2, 1), (2, 1)]):
        super(ClinicalNet, self).__init__()
        # Embedding layer
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y)
                                               for x, y in embedding_dims])

        n_embeddings = sum([y for x, y in embedding_dims])
        n_continuous = 1

        # Linear Layers
        self.fc1 = nn.Linear(n_embeddings + n_continuous, 256)
        self.fc2 = FC(256, output_vector_size, 1)
        self.embedding_dropout = nn.Dropout(p=0.5)

        # Continuous feature batch norm layer
        # self.bn_layer = nn.BatchNorm1d(n_continuous)

        # Output Layer
        n_fc_layers = 4
        n_neurons = 256
        self.lin = FC(in_features=256, out_features=n_neurons, n_layers=n_fc_layers)
        self.lin2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons,
                            out_features=1),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        try:
            categorical_x, continuous_x = x['clinical']
        except TypeError:
            categorical_x, continuous_x = x
        categorical_x = categorical_x.to(torch.int64)

        x = [emb_layer(categorical_x[:, i])
             for i, emb_layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        # x = self.embedding_dropout(x)

        if continuous_x.shape[-1] >1:
            continuous_x = self.bn_layer(continuous_x)

        x = torch.cat([x, continuous_x], 1)
        x = self.fc1(x)
        x_fe = self.fc2(x)
        
        # risk pred
        x = self.lin(x_fe)
        x = self.lin2(x)

        return x, x_fe
    
class miRNA(nn.Module):
    """miRNA data extractor.

    Handle continuous features and categorical feature embeddings.
    """
    def __init__(self):
        super(miRNA, self).__init__()

        # Linear Layers
        output_vector_size = 256
        self.fc1 = nn.Linear(1881, 4096)
        self.fc2 = FC(4096, output_vector_size, 3)
        self.embedding_dropout = nn.Dropout(p=0.5)

        # Continuous feature batch norm layer
        # self.bn_layer = nn.BatchNorm1d(n_continuous)

        # Output Layer
        n_fc_layers = 4
        n_neurons = 256
        self.lin = FC(in_features=output_vector_size, out_features=256, n_layers=n_fc_layers)
        self.lin2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons,
                            out_features=1),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.embedding_dropout(x)
        x_fe = self.fc2(x)
        
        # risk pred
        x = self.lin(x_fe)
        x = self.lin2(x)

        return x, x_fe
    
class GeneNet(nn.Module):
    """GeneNet data extractor.

    Handle continuous features and categorical feature embeddings.
    """
    def __init__(self):
        super(GeneNet, self).__init__()

        # Linear Layers
        output_vector_size = 64
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = FC(64, output_vector_size, 3)
        self.embedding_dropout = nn.Dropout(p=0.5)

        # Continuous feature batch norm layer
        # self.bn_layer = nn.BatchNorm1d(n_continuous)

        # Output Layer
        n_fc_layers = 3
        n_neurons = 128
        self.lin = FC(in_features=output_vector_size, out_features=n_neurons, n_layers=n_fc_layers)
        self.lin2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons,
                            out_features=1),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.embedding_dropout(x)
        x_fe = self.fc2(x)
        
        # risk pred
        x = self.lin(x_fe)
        x = self.lin2(x)

        return x, x_fe

class CnvNet(nn.Module):
    """Gene copy number variation data extractor."""
    def __init__(self, output_vector_size, embedding_dims=[(3, 2)] * 2000):
        super(CnvNet, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y)
                                               for x, y in embedding_dims])
        n_embeddings = 2 * 2000
        self.fc = FC(in_features=n_embeddings, out_features=output_vector_size,
                     n_layers=5, scaling_factor=1)

    def forward(self, x):
        x = x.to(torch.int64)

        x = [emb_layer(x[:, i])
             for i, emb_layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        out = self.fc(x)

        return out

class WsiNet(nn.Module):
    "WSI patch feature extractor and aggregator."
    def __init__(self, output_vector_size):
        super(WsiNet, self).__init__()
        self.feature_extractor = ResNet()
        self.num_image_features = self.feature_extractor.n_features
        # Multiview WSI patch aggregation
        self.fc = FC(self.num_image_features, output_vector_size , 1)

    def forward(self, x):
        view_pool = []

        # Extract features from each patch
        for v in x:
            v = self.feature_extractor(v)
            v = v.view(v.size(0), self.num_image_features)

            view_pool.append(v)

        # Aggregate features from all patches
        patch_features = torch.stack(view_pool).max(dim=1)[0]

        out = self.fc(patch_features)

        return out


class Fusion(nn.Module):
    "Multimodal data aggregator."
    def __init__(self, method, feature_size, device):
        super(Fusion, self).__init__()
        self.method = method
        methods = ['cat', 'max', 'sum', 'prod', 'embrace', 'attention']

        if self.method not in methods:
            raise ValueError('"method" must be one of ', methods)

        if self.method == 'embrace':
            if device is None:
                raise ValueError(
                    '"device" is required if "method" is "embrace"')

            self.embrace = EmbraceNet(device=device)

        if self.method == 'attention':
            if not feature_size:
                raise ValueError(
                    '"feature_size" is required if "method" is "attention"')
            self.attention = Attention(size=feature_size)

    def forward(self, x):
        if self.method == 'attention':
            out = self.attention(x)
        if self.method == 'cat':
            # out = torch.cat([m for m in x], dim=1)
            out = x
        if self.method == 'max':
            out = x.max(dim=0)[0]
        if self.method == 'sum':
            out = x.sum(dim=0)
        if self.method == 'prod':
            out = x.prod(dim=0)
        if self.method == 'embrace':
            out = self.embrace(x)

        return out



import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import SAGPooling
# from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.nn import BatchNorm


class wsi_model(torch.nn.Module):

    def __init__(self):
        super(wsi_model, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)
        self.lin = torch.nn.Linear(256, 256)
        self.lin2 = torch.nn.Linear(256, 1)
        # self.lin3 = torch.nn.Linear(128, 64)
        # self.lin4 = torch.nn.Linear(64, 1)

        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)
        self.norm1 = BatchNorm(512, track_running_stats=False)
        self.norm2 = BatchNorm(512, track_running_stats=False)
        self.norm3 = BatchNorm(256, track_running_stats=False)

    def forward(self, datas):
        # data= data[0]
        wsi_features = []
        for data in datas:
            x = self.conv1(data.x, data.edge_index)
            x = self.norm1(x)
            x = F.relu(x)
            x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

            x = self.conv2(x, edge_index)
            x = self.norm2(x)
            x = F.relu(x)
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

            x = self.conv3(x, edge_index)
            x = self.norm3(x)
            x = F.relu(x)
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

            x = self.conv4(x, edge_index)        
        
            wsi_features.append(global_max_pool(x, batch=batch, size=data.batch.max() + 1))
            
        wsi_feature = torch.stack(wsi_features, dim=0)
        x1 = torch.mean(wsi_feature, dim=0)

        # risk pred
        x = self.lin(x1)
        x = F.relu(x)
        x = self.lin2(x)
        # return x
        return x, x1    

### model.py # ct


def masked_average_pooling(x, mask):
    # x: (4,16,96,160,192)
    # mask: (4,1,96,160,192)
    
    # mask = (y == 1).to(torch.float)
    # mask = y

    eps = 1e-12

    if mask.size()[2:] != x.size()[2:]:
        mask = torch.nn.functional.interpolate(mask, x.size()[2:], mode='trilinear', align_corners=False)

    # area = torch.sum(mask.view(mask.size(0), -1), dim=1, keepdim=True) + eps
    area = torch.sum(mask.reshape(mask.size(0), -1), dim=1, keepdim=True) + eps

    masked = x * mask
    masked = torch.sum(masked.view(masked.size(0), masked.size(1), -1), axis=2)
    masked = masked / area
    return masked

def masked_max_pooling(x, mask):
    """
    Perform max pooling only over the masked region.
    
    Parameters:
    - x: Tensor of shape (B, C, D, H, W)  # Input feature map
    - mask: Tensor of shape (B, 1, D, H, W)  # Binary mask (1 for valid, 0 for invalid)

    Returns:
    - max_pooled: Tensor of shape (B, C) where each channel is max pooled over masked regions
    """
    eps = 1e-12  # Large negative value for masked-out regions

    if mask.size()[2:] != x.size()[2:]:
        mask = torch.nn.functional.interpolate(mask, x.size()[2:], mode='trilinear', align_corners=False)

    # Expand mask to match `x` channels
    mask = mask.expand_as(x)  # (B, C, D, H, W)

    # Apply mask: set unmasked values to x, masked values to a very small number
    masked_x = x * mask + (1 - mask) * eps  # This ensures masked-out values do not interfere with max pooling

    # Max over spatial dimensions (D, H, W)
    max_pooled = torch.amax(masked_x, dim=(2, 3, 4))  # (B, C)

    return max_pooled


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

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act='ReLU'):
        super(SingleConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x
    
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, num_block=1, down_sample=False, up_sample=False, down_op=None, up_op=None, do_residual=True, act_after_res=True):
#         super(ConvBlock, self).__init__()
#         self.up_sample = up_sample
#         self.down_sample = down_sample
#         self.do_residual = do_residual
#         self.act_after_res = act_after_res

#         if up_sample:
#             if up_op == 'transp_conv':
#                 self.upsampling_op = nn.ConvTranspose3d(in_channels, in_channels, kernel_size, 2, 1, output_padding=1)
#             else:
#                 self.upsampling_op = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

#         self.blocks = nn.Sequential()
#         if down_sample:
#             if down_op == 'stride_conv':
#                 self.downsample_op = nn.Conv3d(in_channels, in_channels, kernel_size, stride=2, padding=1)
#                 # self.blocks.append(nn.Conv3d(in_channels, in_channels, kernel_size, stride=2, padding=1))
#             else:
#                 self.downsample_op = nn.AvgPool3d(2)
#         for i in range(num_block):
#             if i > 0:
#                 in_channels_ = out_channels
#             elif up_sample:
#                 in_channels_ = in_channels + out_channels
#             else:
#                 in_channels_ = in_channels
#             self.blocks.append(SingleConv(in_channels_, out_channels, kernel_size))

#         if do_residual:
#             if self.up_sample:
#                 self.skip = nn.Conv3d(in_channels + out_channels, out_channels, 1, bias=False)
#             else:
#                 self.skip = nn.Conv3d(in_channels, out_channels, 1, bias=False)
#             if act_after_res:
#                 self.nonlin = nn.ReLU(inplace=True)

#     def forward(self, x, y=None):
#         if self.up_sample:
#             x = self.upsampling_op(x)
#             x = torch.cat([x, y], dim=1)

#         if self.down_sample:
#             x = self.downsample_op(x)

#         if self.do_residual:
#             residual = self.skip(x)
#             x = self.blocks(x)
#             x = x + residual
#             if self.act_after_res:
#                 x = self.nonlin(x)
#         else:
#             x = self.blocks(x)

#         return x

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
    def __init__(self, nOut=1):
        super(BaseNet, self).__init__()
        nCh = 16
        self.conv1 = ConvBlocks(1, nCh)
        self.conv2 = ConvBlocks(nCh, nCh)
        self.conv3 = ConvBlocks(nCh, nCh)


        nFC = nCh * 4

        self.fc1 = Linear(nFC, nFC)
        # self.fc2 = Linear(nFC, nFC)
        self.fc3 = Linear(nFC, nOut)
        # self.dropout = Dropout(inplace=False)
        self.sigmoid = Sigmoid()

    def forward(self, x): # (B, NAPD, 96,160,192)
        x = x['ct']
        img, mask = x
        phase_feature = []
        for i in range(4):
            img_phase = img[:,i:i+1,:,:,:]
            mask_phase = mask[:,i:i+1,:,:,:]
            img_phase = self.conv1(img_phase) # (B,1,96,160,192) -> (B,16,96,160,192)
            img_phase = self.conv2(img_phase)
            # img_phase = self.conv3(img_phase)
            img_phase = masked_average_pooling(img_phase, mask_phase) # (B,16)
            phase_feature.append(img_phase)
        x_fe = torch.cat(phase_feature,dim=1)

        # x_fe = x.view(1, -1) # (1,64)
        
        # risk pred
        # x = self.dropout(F.relu(self.fc1(x_fe)))
        # x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc1(x_fe))
        # x = F.relu(self.fc2(x))

        risk = self.fc3(x)
        return risk, x_fe


class BaseNet_multi(nn.Module):
    def __init__(self,nOut=1):
        super(BaseNet_multi, self).__init__()
        nCh = 16
        self.conv = ConvBlocks(1, nCh*2)
        self.conv_2 = ConvBlocks(nCh*2, nCh*2, stride=2)

        nFC = nCh * 8
        nFC_2 = nCh * 8
        self.fc1 = Linear(nFC, nFC)
        self.fc2 = Linear(nFC, nFC)
        self.fc3 = Linear(nFC, nFC)

        self.fc1_2 = Linear(nFC_2, nFC_2)
        self.fc2_2 = Linear(nFC_2, nFC_2)
        self.fc3_2 = Linear(nFC_2, nFC_2)

        self.dropout = Dropout()
        self.softmax = nn.Softmax(dim=1)
        self.relu = ReLU()

        self.lin = Linear(nFC+nFC_2, (nFC+nFC_2)//2)
        self.lin2 = Linear((nFC+nFC_2)//2, 32)
        self.lin3 = Linear(32, nOut)

    def forward(self, x):
        # x = x['ct']
        x,y = x
        phase_feature1 = []
        phase_feature2 = []
        
        for i in range(4):
            x_phase = x[:,i:i+1,:,:,:]
            y_phase = y[:,i:i+1,:,:,:]
            conv_q = self.conv(x_phase) # (4,1,96,160,192) -> (4,16,96,160,192)
            v = masked_average_pooling(conv_q, y_phase) # (4,16)
            phase_feature1.append(v)
            
            conv_q_2 = self.conv_2(conv_q) # (4,8,96,160,192) -> # (4,16,96,160,192)
            v_2 = masked_average_pooling(conv_q_2, y_phase) # (4,32)
            phase_feature2.append(v_2)
            
        x_fe1 = torch.cat(phase_feature1,dim=1) # (4,64)
        x_fe2 = torch.cat(phase_feature2,dim=1) # (4,128)

        # x = self.dropout(self.relu(self.fc1(x_fe1)))
        # x = self.dropout(self.relu(self.fc2(x)))
        x = F.relu(self.fc1(x_fe1))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x_2 = F.relu(self.fc1_2(x_fe2))
        x_2 = F.relu(self.fc2_2(x_2))
        x_2 = self.fc3_2(x_2)

        fused_feature = torch.cat([x, x_2], dim=1)
        
        x_fused = F.relu(self.lin(fused_feature))
        x_fused = F.relu(self.lin2(x_fused))
        x_final = self.lin3(x_fused)

        return x_final, fused_feature # , x_final


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
    

### Fusion model

class Fusion_model(nn.Module):
    def __init__(self, modalities):
        super(Fusion_model, self).__init__()
        
        in_channel = 0
        n_Ch = 256
        if 'clinical' in modalities:
            in_channel+=256
        if 'miRNA' in modalities:
            in_channel+=256
        if 'wsi' in modalities:
            in_channel+=256
        if 'ct' in modalities:
            in_channel+=64
        
        self.lin1 = nn.Linear(in_channel, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.concat(x, dim=1)
        x = F.relu(self.lin1)
        x = F.relu(self.lin2)
        x = F.relu(self.lin3)
        out = F.relu(self.lin4)
        return out