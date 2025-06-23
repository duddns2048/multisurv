import os
import torch
import math
import numpy as np


###BLCA
# interval_cut = [0, 20, 57, 82, 110, 163, 191, 220, 251, 272, 294, 330, 344, 376, 394, 428, 466, 481, 495, 524, 544,
                # 577, 602, 630, 649, 696, 696, 758, 812, 832, 893, 945, 997, 1004, 1072, 1423, 1460, 1670, 1804,
                # 1884, 1971]
###GBM
# interval_cut = [0, 3, 22, 41, 82, 110, 123, 146, 177, 204, 218, 225, 232, 256, 277, 310, 323, 342, 369, 375, 386, 430,
#                 442, 454, 458, 476, 486, 541, 598, 620, 690, 705, 747, 790, 812, 845, 975, 1062, 1179, 1339, 1615, 1987]
###BRCA
# interval_cut = [0, 1, 11, 78, 224, 304, 345, 368, 383, 403, 426, 448, 477, 508, 532, 558, 579, 614, 639, 683, 742, 777, 899, 965, 1007, 1093, 1165, 1309, 1471, 1550, 1673, 1866, 2012, 2222, 2372, 2520, 2770, 3001, 3203, 3506, 3957]
###LUSC
# interval_cut = [0, 12, 30, 59, 85, 122, 150, 211, 247, 311, 358, 378, 423, 448, 515, 570, 616, 660, 699, 759, 835, 911, 974, 1058, 1143, 1260, 1426, 1602, 1784, 1927, 2026, 2165, 2409, 2471, 2820, 2979, 3600, 4053, 4601, 5287]
###LUSC within4000
#interval_cut = [0, 12, 28, 52, 61, 105, 131, 153, 211, 247, 306, 353, 371, 405, 429, 474, 517, 573, 616, 653, 699, 734, 818, 881, 923, 983, 1067, 1143, 1223, 1386, 1519, 1690, 1852, 1927, 2080, 2170, 2409, 2447, 2803, 2979, 2979]

###KIRC
interval_cut = [3, 23, 91, 182, 306, 365, 407, 482, 552, 643, 697, 758, 848, 951, 1018, 1122, 1140, 1239, 1314, 1393, 1447, 1491, 1519, 1590, 1676, 1782, 1854, 1888, 1949, 2023, 2184, 2270, 2422, 2552, 2747, 2880, 3269, 3440, 3764, 4537]

num_of_interval = len(interval_cut)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_pred_label(out):
    return torch.argmax(out,dim=2)

def cox_loss(preds,labels,status):
    labels = labels.unsqueeze(1)
    status = status.unsqueeze(1)

    mask = torch.ones(labels.shape[0],labels.shape[0]).cuda()

    mask[(labels.T - labels)>0] = 0 # 어떤 환자의 생존 시간이 다른 환자보다 긴 경우

    clipped_preds = torch.clamp(preds, max=20)
    log_loss = torch.exp(clipped_preds)*mask
    log_loss = torch.sum(log_loss,dim = 0)
    log_loss = torch.log(log_loss).reshape(-1,1)
    log_loss = -torch.sum((clipped_preds-log_loss)*status)

    return log_loss

def time_cox_loss(preds,labels,status):
    labels = labels.unsqueeze(1)
    status = status.unsqueeze(1)

    mask = torch.ones(labels.shape[0],labels.shape[0],dtype=torch.float).cuda()

    mask[(labels.T - labels)>0] = 0
    
    risk_set_size = mask.sum(dim=1)

    log_loss = torch.exp(preds)*mask
    log_loss = torch.sum(log_loss,dim = 0)
    log_loss = torch.log(log_loss).reshape(-1,1)
    log_loss = -torch.mean((preds-log_loss)*status)

    return log_loss


def binary_label(label_l):
    survival_vector = torch.zeros(len(label_l),len(interval_cut))
    for i in range(len(label_l)):
        for j in range(len(interval_cut)):
            if label_l[i]>interval_cut[j]:
                survival_vector[i,j] = 1
    return survival_vector

def binary_last_follow(label_l):

    label_vector = torch.zeros((len(label_l),len(interval_cut)))
    for i in range(len(label_l)):
        for j in range(len(interval_cut)):
            if label_l[i] > interval_cut[j]:
                label_vector[i,j] = 1
            else:
                label_vector[i,j] = -1
    return label_vector

def binary_last_follow_zero(label_l):

    label_vector = torch.zeros((len(label_l),len(interval_cut)))
    for i in range(len(label_l)):
        for j in range(len(interval_cut)):
            if label_l[i] > interval_cut[j]:
                label_vector[i,j] = 1
            else:
                label_vector[i,j] = 0
    return label_vector

def calculate_MAE_with_prob(b_pred, pred, label,status,last_follow):  ###b_pred N*I   label N

    interval = torch.zeros(len(interval_cut)).cuda()

    for i in range(len(interval_cut)):
        if i == 0:
            interval[i] = interval_cut[i+1]
        else:
            interval[i] = interval_cut[i] - interval_cut[i - 1]

    pred = pred.permute(1, 0, 2)
    estimated = torch.mul(b_pred.cuda(), pred[:, :, 1]).cuda()
    observed = torch.mul(last_follow,1-status)+torch.mul(label,status)

    estimated = torch.sum(torch.mul(estimated, interval), dim=1).cuda()

    compare = torch.zeros(len(estimated)).cuda()
    compare_invers = torch.zeros(len(estimated)).cuda()

    for i in range(len(compare)):
        compare[i] = observed[i] > estimated[i]
        compare_invers[i] = observed[i]<=estimated[i]

    MAE = torch.mul(compare,observed-estimated)+torch.mul(torch.mul(status,compare_invers),estimated-observed)


    return torch.sum(MAE)

def calculate_time(b_pred):
    pred_ = torch.zeros(len(b_pred), dtype=float).to(device)

    for i in range(len(pred_)):
        idx = (b_pred[i] == 1).nonzero().squeeze(1)
        if len(idx) == 0:
            idx = torch.zeros(1)
        if int(idx.max().item())<len(interval_cut)-1:

            # pred_[i] = ((interval_cut[int(idx.max().item() )]+interval_cut[int(idx.max().item() + 1)])/2)
            pred_[i] = ((interval_cut[int(idx.max().item() )]+(interval_cut[int(idx.max().item() + 1)]-interval_cut[int(idx.max().item())])/2))
            # pred_[i] = interval_cut[int(idx.max().item() +1)]
        else:
            pred_[i] = (interval_cut[-1]+5)
    return pred_

def calculate_time_prob(b_pred, pred):
    # pred_ = torch.zeros(len(b_pred), dtype=float).to(device)
    interval = torch.zeros(len(interval_cut)).to(device)

    for i in range(len(interval_cut)):
        if i == 0:
            interval[i] = interval_cut[i+1]
        else:
            interval[i] = interval_cut[i] - interval_cut[i - 1]

    pred = pred.permute(1, 0, 2)
    estimated = torch.mul(b_pred.cuda(), pred[:, :, 1]).cuda()
    estimated = torch.sum(torch.mul(estimated, interval), dim=1).cuda()

    return estimated

def calculate_MAE(b_pred, label,status,last_follow):  ###b_pred N*I   label N

    pred_ = torch.zeros(len(label), dtype=float).to(device)

    for i in range(len(label)):
        idx = (b_pred[i] == 1).nonzero().squeeze(1)
        # print(len(idx))
        if len(idx) == 0:
            idx = torch.zeros(1)
        if int(idx.max().item()) < len(interval_cut) - 1:

            pred_[i] = ((interval_cut[int(idx.max().item() )]+interval_cut[int(idx.max().item() + 1)])/2)
            # pred_[i] = interval_cut[int(idx.max().item() +1)]
        else:
            pred_[i] = (interval_cut[-1] + 5)

    observed = torch.mul(last_follow,1-status)+torch.mul(label,status)
    compare = torch.zeros(len(pred_)).cuda()
    compare_invers = torch.zeros(len(pred_)).to(device)
    for i in range(len(compare)):
        compare[i] = observed[i] > pred_[i]
        compare_invers[i] = observed[i]<=pred_[i]

    MAE = torch.mul(compare,observed-pred_)+torch.mul(torch.mul(status,compare_invers),pred_-observed)

    return torch.sum(MAE)

def cross_entropy_all(b_label, pred,status,b_last_follow,weight = 1,cost = 'False'):    ###I * N
    criterion = torch.nn.CrossEntropyLoss(ignore_index= -1,reduction='none')
    total_loss = torch.zeros((num_of_interval,len(pred[0]))).to(device)

    status_ = status.unsqueeze(1)
    b_label = b_label.permute(1,0)
    weight_matrix = torch.zeros(b_label.shape).to(device)

    b_last_follow_ = b_last_follow.permute(1,0)

    combined = torch.mul(b_label,status_)+torch.mul(b_last_follow_,1-status_)
    combined = combined.permute(1,0).to(device).to(torch.long)
    for i in range(len(b_label)):
        a = torch.arange(0,len(weight_matrix[i])).to(device)
        try:
            idx = (combined.permute(1,0)[i] == 1).nonzero().max()
        except:
            idx = torch.zeros(1).to(torch.int).to(device)
        weight_matrix[i] = torch.abs(a-idx)
    for i in range(num_of_interval):
        loss = criterion(pred[i], combined[i])
        if cost == 'True':
            total_loss[i] = loss*weight
        else:
            total_loss[i] = loss
    total_loss = total_loss*weight_matrix.permute(1,0)
    total_loss = torch.sum(total_loss)

    return total_loss


def cross_entropy_all_2(b_label, pred, status, b_last_follow, weight=1):  ###I * N
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none',size_average=False)
    # criterion = torch.nn.CrossEntropyLoss( reduction='none')
    total_loss = torch.zeros((num_of_interval, len(pred[0]))).to(device)

    status_ = status.unsqueeze(1)
    b_label = b_label.permute(1, 0)
    weight_matrix = torch.zeros(b_label.shape).to(device)

    b_last_follow_ = b_last_follow.permute(1, 0)

    combined = torch.mul(b_label, status_) + torch.mul(b_last_follow_, 1 - status_)
    combined = combined.permute(1, 0).to(device).to(torch.long)
    for i in range(len(b_label)):
        a = torch.arange(0, len(weight_matrix[i])).to(device)
        try:
            idx = (combined.permute(1, 0)[i] == 1).nonzero().max()
        except:
            idx = 0
        weight_matrix[i] = torch.abs(a - idx)
    pred = pred.permute(1,2,0)
    # pred = torch.softmax(pred,dim = 1)
    combined = combined.permute(1,0)
    loss = criterion(pred,combined)*weight_matrix
    loss = torch.sum(loss,dim = 1)
    loss = torch.sum(loss,dim = 0)

    # for i in range(num_of_interval):
    #     # importance = math.log(i+3)
    #     a = pred[i]
    #     b = combined[i]
    #     loss = criterion(pred[i], combined[i])
    #
    #     total_loss[i] = loss * weight
    # total_loss = total_loss * weight_matrix.permute(1, 0)
    # total_loss = torch.sum(total_loss)

    return torch.mean(loss)

def calculate_time_MAE(pred_,label,status_):  ###b_pred N*I   label N

    compare = torch.zeros(len(pred_)).to(device)
    compare_invers = torch.zeros(len(pred_)).to(device)
    for i in range(len(compare)):
        compare[i] = label[i] > pred_[i]
        compare_invers[i] = label[i]<=pred_[i]
    MAE = torch.mul(compare,label-pred_)+torch.mul(torch.mul(status_,compare_invers),pred_-label)

    return torch.sum(MAE)


