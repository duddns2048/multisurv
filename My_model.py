import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import SAGPooling
from torch_sparse import SparseTensor
import numpy as np
from torch_geometric.nn import BatchNorm


class BinaryClassifier(torch.nn.Module):

    def __init__(self, classifier_in):
        super(BinaryClassifier, self).__init__()
        in_channels = classifier_in
        out_channels = 2

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=out_channels),
        )

    def forward(self, x):
        return self.classifier(x)


class BinaryClassifier_2fc(torch.nn.Module):

    def __init__(self, classifier_in):
        super(BinaryClassifier_2fc, self).__init__()
        in_channels = classifier_in
        out_channels = 2

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_channels, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=out_channels)
        )

    def forward(self, x):
        return self.classifier(x)


class BigModel(torch.nn.Module):

    def __init__(self, num_of_interval, classifier_in):
        super(BigModel, self).__init__()
        self.interval = num_of_interval
        self.num_class = 2
        for i in range(self.interval):
            exec('self.classifier%s= BinaryClassifier(classifier_in)' % i)

    def forward(self, x):
        predictions = torch.zeros((self.interval, len(x), self.num_class)).cuda()
        for i in range(self.interval):
            exec('x%s = self.classifier%s(x)' % (i, i))
            exec('predictions[%s] = x%s' % (i, i))
        predictions = predictions.permute(1, 0, 2)
        return predictions


class BigModel_onevector(torch.nn.Module):

    def __init__(self, num_of_interval, classifier_in):
        super(BigModel_onevector, self).__init__()
        self.interval = num_of_interval
        self.num_class = 2
        self.fc = torch.nn.Linear(classifier_in, self.interval * self.num_class)

    def forward(self, x):
        x = self.fc(x)

        return x


class Ranking_ordinal(torch.nn.Module):

    def __init__(self, num_of_interval, classifier_in):
        super(Ranking_ordinal, self).__init__()

        self.fc = torch.nn.Linear(classifier_in, 128, bias=False)
        self.fc2 = torch.nn.Linear(128, 1, bias=False)
        self.linear_1_bias = torch.nn.Parameter(torch.zeros(num_of_interval).float())

    def forward(self, x):
        logits = self.fc(x)
        logits = F.relu(logits)
        logits = self.fc2(logits)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)

        return logits, probas


class GCN_Risk(torch.nn.Module):

    def __init__(self):
        super(GCN_Risk, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)
        self.lin = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()
        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x = self.conv4(x, edge_index)

        x = torch.cat((global_max_pool(x, batch=batch, size=data.batch.max() + 1),
                       global_mean_pool(x, batch=batch, size=data.batch.max() + 1)), dim=1)
        x = self.lin(x)

        return x


class GCN_Time_norm(torch.nn.Module):

    def __init__(self, num_interval, classifier_in):
        super(GCN_Time_norm, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)

        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)

        self.norm1 = BatchNorm(512, track_running_stats=False)
        self.norm2 = BatchNorm(512, track_running_stats=False)
        self.norm3 = BatchNorm(256, track_running_stats=False)


    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.norm1(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x = self.conv4(x, edge_index)

        x = torch.cat((global_max_pool(x, batch=batch, size=data.batch.max() + 1),
                       global_mean_pool(x, batch=batch, size=data.batch.max() + 1)), dim=1)


        return x

class GCN_Time_Only_norm(torch.nn.Module):

    def __init__(self, num_interval, classifier_in):
        super(GCN_Time_Only_norm, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)

        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)

        self.norm1 = BatchNorm(512, track_running_stats=False)
        self.norm2 = BatchNorm(512, track_running_stats=False)
        self.norm3 = BatchNorm(256, track_running_stats=False)
        self.classifier = BigModel(num_of_interval=num_interval, classifier_in=classifier_in)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = self.norm1(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.tanh(x)

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x = self.conv4(x, edge_index)

        x = torch.cat((global_max_pool(x, batch=batch, size=data.batch.max() + 1),
                       global_mean_pool(x, batch=batch, size=data.batch.max() + 1)), dim=1)
        x = self.classifier(x)


        return x


class GCN_BIG(torch.nn.Module):

    def __init__(self, num_interval, ordinal_model):
        super(GCN_BIG, self).__init__()
        classifier_in = 512
        # self.GCN_cox = GCN_Risk()
        self.GCN_ordinal = ordinal_model
        self.lin = torch.nn.Linear(512, 1)
        # self.classifier = BigModel(num_of_interval=num_interval, classifier_in=classifier_in)

    def forward(self, x):
        # risk = self.GCN_cox(x)
        # risk = self.lin(cox_feature)
        ordinal_result, ordinal_feature = self.GCN_ordinal(x)
        # risk_feature = ordinal_feature.reshape(len(ordinal_feature),-1,1).squeeze(2)
        # risk_feature = self.GCN_ordinal(x)
        risk = self.lin(ordinal_feature)
        # ordinal_feature = torch.cat((ordinal_feature,cox_feature.detach()),dim = 1)
        # ordinal_feature = torch.cat((ordinal_feature.detach(), risk), dim=1)
        # x = self.classifier(ordinal_feature)

        return ordinal_result, risk


class GCN_Risk_Only(torch.nn.Module):

    def __init__(self):
        super(GCN_Risk_Only, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)
        self.lin = torch.nn.Linear(512, 256)
        self.lin2 = torch.nn.Linear(256, 1)

        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)
        self.norm1 = BatchNorm(512, track_running_stats=False)
        self.norm2 = BatchNorm(512, track_running_stats=False)
        self.norm3 = BatchNorm(256, track_running_stats=False)

    def forward(self, data):
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

        x1 = torch.cat((global_max_pool(x, batch=batch, size=data.batch.max() + 1),
                        global_mean_pool(x, batch=batch, size=data.batch.max() + 1)), dim=1)

        x = self.lin(x1)
        x = F.relu(x)
        x = self.lin2(x)
        
        return x, x1  # x1: WSI feature

    
class GCN_Time_Only(torch.nn.Module):

    def __init__(self, num_interval, classifier_in):
        super(GCN_Time_Only, self).__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)

        self.relu = torch.nn.ReLU()
        self.pool1 = SAGPooling(512, 0.6)
        self.pool2 = SAGPooling(512, 0.6)
        self.pool3 = SAGPooling(256, 0.5)
        self.classifier = BigModel(num_of_interval=num_interval, classifier_in=classifier_in)



    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, data.edge_index, batch=data.batch)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, batch=batch)

        x2 = self.conv4(x, edge_index)

        x1 = torch.cat((global_max_pool(x2, batch=batch, size=data.batch.max() + 1),
                        global_mean_pool(x2, batch=batch, size=data.batch.max() + 1)), dim=1)

        return x1



class GCN_Freeze_Risk(torch.nn.Module):

    def __init__(self, num_interval):
        super(GCN_Freeze_Risk, self).__init__()
        classifier_in = 512

        self.GCN_cox = GCN_Risk_Only()
        self.GCN_ordinal = GCN_Time_norm(num_interval=num_interval, classifier_in=classifier_in)

        # self.lin2 = torch.nn.Linear(1024,512)

        self.classifier = BigModel(num_of_interval=num_interval, classifier_in=classifier_in)

    def forward(self, x, train='True'):

        r, r_feature = self.GCN_cox(x)

        x = self.GCN_ordinal(x)     ### N*512
        r_feature = r_feature.detach_()
        r = r.detach_()
        # x = torch.cat((x, r_feature), dim=1)  #### N*1024
        x = torch.add(x,r_feature)
        x = x.div(2)

        # x = self.lin2(x)

        x = self.classifier(x)

        return x, r



class model2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        in_chan = 512
        self.conv1 = GCNConv(in_chan, 512, add_self_loops=False)
        self.bn1 = BatchNorm(512, track_running_stats=False)
        # self.norm1 = torch.nn.BatchNorm1d(1024)
        # self.pool1 = SAGPooling(512, 0.6, GNN=GCNConv)
        self.pool1 = SAGPooling(512, 0.6)
        self.conv2 = GCNConv(512, 512, add_self_loops=False)
        self.bn2 = BatchNorm(512, track_running_stats=False)
        # self.pool2 = SAGPooling(512, 0.6, GNN=GCNConv)
        self.pool2 = SAGPooling(512, 0.6)
        self.conv3 = GCNConv(512, 256, add_self_loops=False)
        self.bn3 = BatchNorm(256, track_running_stats=False)
        # self.pool3 = SAGPooling(256, 0.5, GNN=GCNConv)
        self.pool3 = SAGPooling(256, 0.5)
        self.conv4 = GCNConv(256, 256, add_self_loops=False)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(512, 1)
        # self.fc2 = nn.Linear(128,1)

    def forward(self, data):
        batch_size = len(data.y)
        out = self.conv1(data.x, data.edge_index)
        out = self.bn1(out)
        out = self.relu(out)
        out, edge, _, batch, _, _ = self.pool1(out, data.edge_index, batch=data.batch)
        out = self.conv2(out, edge)
        out = self.bn2(out)
        out = self.relu(out)
        out, edge, _, batch, _, _ = self.pool2(out, edge, batch=batch)
        out = self.conv3(out, edge)
        out = self.bn3(out)
        out = self.relu(out)
        out, edge, _, batch, _, _ = self.pool3(out, edge, batch=batch)
        out = self.conv4(out, edge)
        out1 = global_max_pool(out, batch, size=batch_size)
        out2 = global_mean_pool(out, batch, size=batch_size)
        feature = torch.cat((out1, out2), 1)
        out = self.fc1(feature)

        return out, feature
