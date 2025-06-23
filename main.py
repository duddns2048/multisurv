import argparse
import json
import os
import random

import numpy as np
import torch
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader as utils_DataLoader
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch_geometric.data import Data
import torch.nn as nn

from My_datasets import unified_dataset2
from multisurv import MultiSurv, MultiSurv2
from Utils_KIRC import interval_cut, cox_loss, time_cox_loss

import nibabel as nib

import matplotlib.pyplot as plt
import sys

def plot_graph(save_dir, train_losses, valid_losses, plot):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=f'Train {plot}')
    plt.plot(valid_losses, label=f'Valid {plot}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{plot}')
    plt.title(f'Training and Validation {plot} per Epoch')
    plt.legend()
    plt.grid(True)

    # 그래프를 PNG 파일로 저장
    plt.savefig(save_dir+f'/{plot}_graph.png')
    plt.close()
    
    
def main(exp_name, label_path, train_npz_dir, test_npz_dir, output_dir, epochs, modalities):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, exp_name), exist_ok=True)
    output_dir = os.path.join(output_dir, exp_name)
    
    with open(os.path.join(output_dir,'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
        
    command_line_arguments = ' '.join(sys.argv)
    with open(os.path.join(output_dir, 'command.txt'), 'w') as f:
        f.write(command_line_arguments)

    train_data = unified_dataset2(train_npz_dir, label_path, modalities, device)
    train_batch_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_data = unified_dataset2(test_npz_dir,label_path, modalities, device)
    test_batch_data  = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # ct
    # train_data = CT_dataset(train_npz_dir, label_path, modalities, device)
    # train_batch_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # test_data = CT_dataset(test_npz_dir, label_path, modalities, device)
    # test_batch_data = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    print("="*100)
    print(f"Total: {len(train_data)+len(test_data)} cases | Train: {len(train_data)} cases | Test: {len(test_data)} cases ")
        
    ## model
    model = MultiSurv2(modalities, finetune=True)

    # CT
    # model = BaseNet()
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    train_loss_list = []
    train_risk_list = []
    val_loss_list = []
    val_risk_list = []
    risk_highest = 0.0
    for epoch in range(epochs):
        print(f'========== Epoch {epoch} ===========================================================')
        model.train()
        
        train_loss=0.0
        
        risk_list = []
        y_list = []
        status_list = []
        
        for train_batch in tqdm(train_batch_data, total=len(train_batch_data), desc=f'Training Epoch {epoch}'):
            optimizer.zero_grad()
            data_dict, label, case_name = train_batch
            status, y = label['event'], label['time']
            
            risk = model(data_dict)
            
            loss = time_cox_loss(risk, y, status)
            loss.backward()
            train_loss += loss.item()
            
            y_list.extend(y.cpu().detach().tolist())
            risk_list.extend(risk.cpu().detach().tolist())
            status_list.extend(status.cpu().detach().tolist())
            
            optimizer.step()
        
        train_risk = concordance_index(np.array(y_list), -np.array(risk_list), np.array(status_list))
        
        train_loss_list.append(train_loss)
        train_risk_list.append(train_risk)

        print(f'Train Loss: {train_loss:.6f} | Train Risk-index: {train_risk:.6f}')
        
        model.eval()
        val_loss = 0.0
        y_list = []
        risk_list = []
        status_list = []
        with torch.no_grad():
            for val_batch in test_batch_data:
                data_dict, label, case_name = val_batch
                status, y = label['event'], label['time']
                
                risk = model(data_dict)                
                loss = time_cox_loss(risk, y, status)
                val_loss += loss.item()
                
                y_list.extend(y.cpu().detach().tolist())
                risk_list.extend(risk.cpu().detach().tolist())
                status_list.extend(status.cpu().detach().tolist())
                
            val_risk = concordance_index(np.array(y_list), -np.array(risk_list), np.array(status_list))
            
            val_loss_list.append(val_loss)
            val_risk_list.append(val_risk)
            
        print(f'Val   Loss: {val_loss:.6f} | Val   Risk-index: {val_risk :.6f}')
        
        plot_graph(output_dir, train_loss_list, val_loss_list,'loss')
        plot_graph(output_dir, val_loss_list, val_loss_list,'val loss') 
        plot_graph(output_dir, train_risk_list, val_risk_list,'C-index')
        state_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
        torch.save(state_dict, os.path.join(output_dir, f'latest_checkpoint.pth'))
        if val_risk > risk_highest:
            risk_highest = val_risk
            torch.save(state_dict, os.path.join(output_dir, f'best_model.pth'))
            
        print(f'Highest Risk-index: {risk_highest:.6f}')
        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='wsi_clinical_tmp', help='experiment name')
    parser.add_argument('--label_path', type=str, default='./TCGA-KIRC/labels_537.tsv', help='experiment name')
    parser.add_argument('--train_npz_dir', type=str, default='./TCGA-KIRC/KIRC_npz/train', help='training set dir')
    parser.add_argument('--test_npz_dir', type=str, default='./TCGA-KIRC/KIRC_npz/test', help='test set dir')
    parser.add_argument('--log_dir', type=str, default='./runs/KIRC/', help='log directory')
    parser.add_argument('--output_dir', type=str, default='./trained/KIRC', help='model save dir')
    # parser.add_argument('--best_ckpt', type=str, default='./trained/KIRC/best_ckpt', help='best ckpt dir')
    parser.add_argument('--seed', type=int, default=1024, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=64, help='mini_batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')

    args = parser.parse_args()
    
    args_dict = vars(args)

    modalities = ['wsi', 'clinical'] # clinical, miRNA, wsi, 
    args_dict['modalities'] = modalities
    
    main(
        exp_name = args.exp_name,
        label_path = args.label_path,
        train_npz_dir = args.train_npz_dir,
        test_npz_dir = args.test_npz_dir,
        output_dir = args.output_dir,
        epochs = args.epochs, 
        modalities = modalities,
        )
    