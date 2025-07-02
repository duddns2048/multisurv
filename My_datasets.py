import torch
from torch.utils.data import Dataset as utils_Dataset
import os
from torch_geometric.data import Dataset,Data,InMemoryDataset,DataLoader
import numpy as np
import csv
import pandas as pd
import nibabel as nib

class wsi_dataset(Dataset):
    def __init__(self, base_dir, label_path, modalities, device):
        super(wsi_dataset,self).__init__()
        self.base_dir = base_dir
        self.data_dirs = os.path.join(base_dir, 'wsi')
        self.label = pd.read_csv(label_path, sep='\t')
        group = base_dir.split('/')[-1]
        self.submitter_ids = list(self.label[self.label['group']==group]['submitter_id'])
        self.device = device
        
    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data

    def _get_wsi(self, data_dirs, case_name):
        data_file_lists = os.listdir(data_dirs)
        file_names = [file for file in data_file_lists if file.startswith(case_name)]
        wsi_data_list = []
        
        for i, file_name in enumerate(file_names):
            file = np.load(os.path.join(data_dirs,file_name),allow_pickle=True)
            feature = file['features']
            edge_index = torch.tensor(file['edge_index'].astype(np.int64))
                
            data = Data(x=torch.Tensor(feature).to(self.device), edge_index=edge_index.to(self.device))
            wsi_data_list.append(data)
        return wsi_data_list
            
    def get(self,index):
        case_name = self.submitter_ids[index]
        data = self._get_wsi(self.data_dirs, case_name)
        
        label ={}
        label['time'] = self.label.loc[self.label['submitter_id'] == case_name, 'time'].values[0]
        label['time'] = torch.tensor(label['time'], dtype = float)
        label['event'] = self.label.loc[self.label['submitter_id'] == case_name, 'event'].values[0]
        label['event'] = torch.tensor(label['event'], dtype = float)
        label = self._data_to_device(label)
        
        return data, label, case_name
        
    def len(self):
        return len(self.submitter_ids)
        
    def get_old(self,index):
        path_dir = self.dir[index]
        file = np.load(path_dir,allow_pickle=True)
        patient_id = file['patient_id'][0]
        feature = file['features']
        edge_index = torch.tensor(file['edge_index'].astype(np.int64))
        status = file['status']
        try:
            follow = file['last_follow_up']
        except:
            follow = 0
        survival_time = file['survival_time']
        # preds = torch.as_tensor(file['preds'])
        # avg = torch.as_tensor(file['average'])
        surv_time = torch.as_tensor(survival_time)
        if status == 0:
            last_follow_up = torch.as_tensor(follow)
        else:
            last_follow_up = torch.as_tensor(survival_time)
        if status == 0:
            out_status = torch.zeros(1)
            time = torch.as_tensor(follow)
        else:
            out_status = torch.ones(1)
            time = torch.from_numpy(survival_time)
        
        data = Data(x=torch.Tensor(feature), edge_index=edge_index, y=time, status=out_status,surv_time =surv_time,last_follow_up =last_follow_up,patient_id=patient_id)
        return data


class CT_dataset(utils_Dataset):
    def __init__(self,base_dir, label_path, modalities, device):
        self.base_dir = base_dir
        self.device = device
        self.label = pd.read_csv(label_path, sep='\t')
        group = base_dir.split('/')[-1]
        self.submitter_ids = list(self.label[self.label['group']==group]['submitter_id'])
                
    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data
        
    def _get_ct(self, data_dir, patient_id, folder):
        ct_data_dir = os.path.join(data_dir,folder)
        ct_case_dir = os.path.join(ct_data_dir, patient_id)
        if not os.path.exists(ct_case_dir):
            ct_data_list_ = torch.zeros((4,96,160,192), device=self.device)
        else:
            ct_data_list = [None, None,None,None] # N A P D
            
            case_file_lists = os.listdir(ct_case_dir)
            
            for file_name in case_file_lists:
                file = nib.load(os.path.join(ct_case_dir, file_name))
                data = file.get_fdata()
                mod = file_name.split('_')[1][0] # N A P D
                
                target_shape = (96, 160, 192)
                padding = []

                for i in range(3):
                    current_dim = data.shape[i] if i < len(data.shape) else 1
                    pad_size = max(0, target_shape[i] - current_dim)
                    pad_before = pad_size // 2
                    pad_after = pad_size - pad_before
                    padding.append((pad_before, pad_after))
                
                data = np.pad(data, padding, mode='constant')
                
                mod_index = {'N': 0, 'A': 1, 'P': 2, 'D': 3}.get(mod, None)
                if mod_index is not None:
                    # 텐서로 변환하고, 리스트에 저장
                    ct_data_list[mod_index] = torch.tensor(data, device=self.device, dtype=torch.float32).unsqueeze(0)
                    
            for i in range(4):
                if ct_data_list[i] is None:
                    # 이미 존재하는 데이터의 shape를 기준으로 0으로 채운 텐서 생성
                    dummy_shape = (1, 96, 160, 192) # (채널, 첫 번째 차원, 두 번째 차원, 세 번째 차원)
                    ct_data_list[i] = torch.zeros(dummy_shape, device=self.device, dtype=torch.float32)
            # ct_data_list_ = torch.stack(ct_data_list,dim=0)
            ct_data_list_ = torch.cat(ct_data_list,dim=0)
        
        return ct_data_list_
    
    def __getitem__(self, index):
        case_name = self.submitter_ids[index]
        
        ct_data = self._get_ct(self.base_dir, case_name, 'ct')
        ct_seg_data = self._get_ct(self.base_dir, case_name, 'ct_seg')
        
        label ={}
        label['time'] = self.label.loc[self.label['submitter_id'] == case_name, 'time'].values[0]
        label['time'] = torch.tensor(label['time'], dtype = float)
        label['event'] = self.label.loc[self.label['submitter_id'] == case_name, 'event'].values[0]
        label['event'] = torch.tensor(label['event'], dtype = float)
        label = self._data_to_device(label)
        return (ct_data, ct_seg_data), label, case_name
    
    def __len__(self):
        return len(self.submitter_ids)
    
class unified_dataset_cox(Dataset):
    def __init__(self, dir, label_path, modalities, device): # dir = ~/wsi/TCGA-KIRC/train or val or test
        super(unified_dataset_cox,self).__init__()
        
        self.modalities = [mod for mod in modalities if mod not in ['wsi', 'ct']]
        self.device = device
        
        files = os.listdir(os.path.join(dir,'clinical'))
        self.cases = [file[:12] for file in files]
        self.data_dirs = {x:os.path.join(dir, x) for x in modalities}
        self.label = pd.read_csv(label_path, sep='\t')
        # self.dir = sorted([os.path.join(dir,file) for file in files])
        self.data_loading_fns = {'clinical': self._get_clinical,
                                 'miRNA': self._get_miRNA,
                                #  'mRNA': self._get_mRNA,
                                #  'CNV': self._get_CNV,
                                #  'ct': self._get_ct
                                 }
    
    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data
        
    def _get_clinical(self, data_dir, patient_id):
        data_file = os.path.join(data_dir, patient_id+'.tsv')
        data = self._read_patient_file(data_file)
        categorical = torch.tensor(
            [int(float(value)) for value in data[:5]], dtype=torch.int)
        continuous = torch.tensor(
            [float(value) for value in data[5:6]], dtype=torch.float)

        return categorical, continuous
    
    def _read_patient_file(self, path):
        with open(path, 'r') as f:
            f = csv.reader(f, delimiter='\t')

            values = []
            for row in enumerate(f):
                values.append(row[1][0])

        return values
    
    def _get_miRNA(self, data_dir, patient_id):
        patient_files = {os.path.splitext(f)[0]: f
                         for f in os.listdir(data_dir)}

        if patient_id in patient_files:
            data_file = os.path.join(data_dir, patient_files[patient_id])
            data = self._read_patient_file(data_file)
            data = torch.tensor([float(value) for value in data])
        else:  # Return all-zero tensor if data is missing
            eg_file = os.path.join(data_dir, list(patient_files.values())[0])
            nfeatures = len(self._read_patient_file(eg_file))
            data = torch.zeros(nfeatures)

        return data
    
    def _get_mRNA(self, data_dir, patient_id):
        return
    def _get_CNV(self, data_dir, patient_id):
        return
    
    def __getitem__(self,index):
        case_name = self.cases[index]
        
        data = {mod: self.data_loading_fns[mod](self.data_dirs[mod],case_name) for mod in self.modalities}
        data = self._data_to_device(data)
        
        label ={}
        label['time'] = self.label.loc[self.label['submitter_id'] == case_name, 'time'].values[0]
        label['time'] = torch.tensor(label['time'], dtype = float)
        label['event'] = self.label.loc[self.label['submitter_id'] == case_name, 'event'].values[0]
        label['event'] = torch.tensor(label['event'], dtype = float)
        label = self._data_to_device(label)
        
        return data, label, case_name
    
    def get(self,index):
        return self.__getitem__(index)
    
    def len(self):
        return len(self.cases)
    

    
class dataset_survive_binary(Dataset):
    def __init__(self,dir):
        super(dataset_survive_binary,self).__init__()
        files = os.listdir(dir)
        self.dir = [os.path.join(dir,file) for file in files]

    def get(self,index):
        path_dir = self.dir[index]
        file = np.load(path_dir,allow_pickle=True)
        patient_id = file['patient_id'][0]
        feature = file['features']
        edge_index = torch.tensor(file['edge_index'].astype(np.int64))
        status = file['status']
        try:
            follow = file['last_follow_up']
        except:
            follow = 0
        survival_time = file['survival_time']
        # preds = torch.as_tensor(file['preds'])
        # avg = torch.as_tensor(file['average'])
        surv_time = torch.as_tensor(survival_time)
        if status == 0:
            last_follow_up = torch.as_tensor(follow)
        else:
            last_follow_up = torch.as_tensor(survival_time)
        if status == 0:
            out_status = torch.zeros(1)
            time = torch.as_tensor(follow)
        else:
            out_status = torch.ones(1)
            time = torch.from_numpy(survival_time)
        
        survive_in_5_years = file['survive_in_5_years']
        survive_in_5_years = torch.as_tensor(survive_in_5_years)
            
        data = Data(x=torch.Tensor(feature), edge_index=edge_index, y=survive_in_5_years, status=out_status,surv_time =surv_time,last_follow_up =last_follow_up,patient_id=patient_id)
        return data


    def len(self):
        return len(self.dir)
        
    
class unified_dataset_wo_wsi(utils_Dataset):
    def __init__(self, base_dir, label_path, modalities, device): # dir = ~/wsi/TCGA-KIRC/train or val or test
        super(unified_dataset_wo_wsi,self).__init__()
        
        self.modalities = [mod for mod in modalities if mod not in ['wsi']]
        self.device = device
        
        self.base_dir = base_dir
        # self.submitter_ids = [file[:12] for file in os.listdir(os.path.join(dir,'clinical'))]
        self.data_dirs = {x: os.path.join(base_dir, x) for x in modalities}
        self.label = pd.read_csv(label_path, sep='\t')
        group = base_dir.split('/')[-1]
        self.submitter_ids = list(self.label[self.label['group']==group]['submitter_id'])
        self.data_loading_fns = {'clinical': self._get_clinical,
                                 'miRNA': self._get_miRNA,
                                #  'mRNA': self._get_mRNA,
                                #  'CNV': self._get_CNV,
                                 'ct': self._get_ct
                                 }
    
    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data
    
    
    
    def get_ct(self, data_dir, patient_id, folder):
        ct_data_dir = os.path.join(data_dir,folder)
        ct_case_dir = os.path.join(ct_data_dir, patient_id)
        if not os.path.exists(ct_case_dir):
            ct_data_list_ = torch.zeros((4,96,160,192), device=self.device)
        else:
            ct_data_list = [None, None,None,None] # N A P D
            
            case_file_lists = os.listdir(ct_case_dir)
            
            for file_name in case_file_lists:
                file = nib.load(os.path.join(ct_case_dir, file_name))
                data = file.get_fdata()
                mod = file_name.split('_')[1][0] # N A P D
                
                target_shape = (96, 160, 192)
                padding = []

                for i in range(3):
                    current_dim = data.shape[i] if i < len(data.shape) else 1
                    pad_size = max(0, target_shape[i] - current_dim)
                    pad_before = pad_size // 2
                    pad_after = pad_size - pad_before
                    padding.append((pad_before, pad_after))
                
                data = np.pad(data, padding, mode='constant')
                
                mod_index = {'N': 0, 'A': 1, 'P': 2, 'D': 3}.get(mod, None)
                if mod_index is not None:
                    # 텐서로 변환하고, 리스트에 저장
                    ct_data_list[mod_index] = torch.tensor(data, device=self.device, dtype=torch.float32).unsqueeze(0)
                    
            for i in range(4):
                if ct_data_list[i] is None:
                    # 이미 존재하는 데이터의 shape를 기준으로 0으로 채운 텐서 생성
                    dummy_shape = (1, 96, 160, 192) # (채널, 첫 번째 차원, 두 번째 차원, 세 번째 차원)
                    ct_data_list[i] = torch.zeros(dummy_shape, device=self.device, dtype=torch.float32)
            # ct_data_list_ = torch.stack(ct_data_list,dim=0)
            ct_data_list_ = torch.cat(ct_data_list,dim=0)
        
        return ct_data_list_
    
    def _get_ct(self, data_dir, patient_id):
        ct = self.get_ct(data_dir, patient_id, 'ct')
        ct_seg = self.get_ct(data_dir, patient_id, 'ct_seg')
        return (ct, ct_seg)
    
    def _get_clinical(self, data_dir, patient_id):
        data_file = os.path.join(data_dir, patient_id+'.tsv')
        data = self._read_patient_file(data_file)
        categorical = torch.tensor(
            [int(float(value)) for value in data[:4]], dtype=torch.int)
        continuous = torch.tensor(
            [float(value) for value in data[4:5]], dtype=torch.float)

        return categorical, continuous
    
    def _read_patient_file(self, path):
        with open(path, 'r') as f:
            f = csv.reader(f, delimiter='\t')

            values = []
            for row in enumerate(f):
                values.append(row[1][0])

        return values
    
    def _get_miRNA(self, data_dir, patient_id):
        patient_files = {os.path.splitext(f)[0]: f
                         for f in os.listdir(data_dir)}

        if patient_id in patient_files:
            data_file = os.path.join(data_dir, patient_files[patient_id])
            data = self._read_patient_file(data_file)
            data = torch.tensor([float(value) for value in data])
        else:  # Return all-zero tensor if data is missing
            eg_file = os.path.join(data_dir, list(patient_files.values())[0])
            nfeatures = len(self._read_patient_file(eg_file))
            data = torch.zeros(nfeatures)

        return data
    
    def _get_mRNA(self, data_dir, patient_id):
        return
    def _get_CNV(self, data_dir, patient_id):
        return
    
    def __getitem__(self,index):
        case_name = self.submitter_ids[index]
        
        data = {mod: self.data_loading_fns[mod](self.data_dirs[mod],case_name) for mod in self.modalities}
        data = self._data_to_device(data)
        
        label ={}
        label['time'] = self.label.loc[self.label['submitter_id'] == case_name, 'time'].values[0]
        label['time'] = torch.tensor(label['time'], dtype = float)
        label['event'] = self.label.loc[self.label['submitter_id'] == case_name, 'event'].values[0]
        label['event'] = torch.tensor(label['event'], dtype = float)
        label = self._data_to_device(label)
        
        return data, label, case_name
    
    def get(self,index):
        return self.__getitem__(index)
    
    def __len__(self):
        return len(self.submitter_ids)
    

class unified_dataset(utils_Dataset):
    def __init__(self, base_dir, label_path, modalities, device): # dir = ~/wsi/TCGA-KIRC/train or val or test
        super(unified_dataset,self).__init__()
        
        self.modalities = [mod for mod in modalities if mod not in ['wsi']]
        self.device = device
        
        self.base_dir = base_dir
        # self.submitter_ids = [file[:12] for file in os.listdir(os.path.join(dir,'clinical'))]
        self.data_dirs = {x: os.path.join(base_dir, x) for x in modalities}
        self.label = pd.read_csv(label_path, sep='\t')
        group = base_dir.split('/')[-1]
        self.submitter_ids = list(self.label[self.label['group']==group]['submitter_id'])
        self.data_loading_fns = {'clinical': self._get_clinical,
                                 'miRNA': self._get_miRNA,
                                #  'mRNA': self._get_mRNA,
                                #  'CNV': self._get_CNV,
                                 'ct': self._get_ct,
                                 'wsi': self._get_wsi
                                 }
    
    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data
    
    
    
    def get_ct(self, data_dir, patient_id, folder):
        ct_data_dir = os.path.join(data_dir,folder)
        ct_case_dir = os.path.join(ct_data_dir, patient_id)
        if not os.path.exists(ct_case_dir):
            ct_data_list_ = torch.zeros((4,96,160,192), device=self.device)
        else:
            ct_data_list = [None, None,None,None] # N A P D
            
            case_file_lists = os.listdir(ct_case_dir)
            
            for file_name in case_file_lists:
                file = nib.load(os.path.join(ct_case_dir, file_name))
                data = file.get_fdata()
                mod = file_name.split('_')[1][0] # N A P D
                
                target_shape = (96, 160, 192)
                padding = []

                for i in range(3):
                    current_dim = data.shape[i] if i < len(data.shape) else 1
                    pad_size = max(0, target_shape[i] - current_dim)
                    pad_before = pad_size // 2
                    pad_after = pad_size - pad_before
                    padding.append((pad_before, pad_after))
                
                data = np.pad(data, padding, mode='constant')
                
                mod_index = {'N': 0, 'A': 1, 'P': 2, 'D': 3}.get(mod, None)
                if mod_index is not None:
                    # 텐서로 변환하고, 리스트에 저장
                    ct_data_list[mod_index] = torch.tensor(data, device=self.device, dtype=torch.float32).unsqueeze(0)
                    
            for i in range(4):
                if ct_data_list[i] is None:
                    # 이미 존재하는 데이터의 shape를 기준으로 0으로 채운 텐서 생성
                    dummy_shape = (1, 96, 160, 192) # (채널, 첫 번째 차원, 두 번째 차원, 세 번째 차원)
                    ct_data_list[i] = torch.zeros(dummy_shape, device=self.device, dtype=torch.float32)
            # ct_data_list_ = torch.stack(ct_data_list,dim=0)
            ct_data_list_ = torch.cat(ct_data_list,dim=0)
        
        return ct_data_list_
    
    def _get_ct(self, data_dir, patient_id):
        ct = self.get_ct(data_dir, patient_id, 'ct')
        ct_seg = self.get_ct(data_dir, patient_id, 'ct_seg')
        return (ct, ct_seg)
    
    def _get_clinical(self, data_dir, patient_id):
        data_file = os.path.join(data_dir, patient_id+'.tsv')
        data = self._read_patient_file(data_file)
        categorical = torch.tensor(
            [int(float(value)) for value in data[:4]], dtype=torch.int)
        continuous = torch.tensor(
            [float(value) for value in data[4:5]], dtype=torch.float)

        return categorical, continuous
    
    def _get_wsi(self, data_dirs, case_name):
        data_file_lists = os.listdir(data_dirs)
        file_names = [file for file in data_file_lists if file.startswith(case_name)]
        wsi_data_list = []
        
        for i, file_name in enumerate(file_names):
            file = np.load(os.path.join(data_dirs,file_name),allow_pickle=True)
            feature = file['features']
            edge_index = torch.tensor(file['edge_index'].astype(np.int64))
                
            data = Data(x=torch.Tensor(feature).to(self.device), edge_index=edge_index.to(self.device))
            wsi_data_list.append(data)
        return wsi_data_list
    
    def _read_patient_file(self, path):
        with open(path, 'r') as f:
            f = csv.reader(f, delimiter='\t')

            values = []
            for row in enumerate(f):
                values.append(row[1][0])

        return values
    
    def _get_miRNA(self, data_dir, patient_id):
        patient_files = {os.path.splitext(f)[0]: f
                         for f in os.listdir(data_dir)}

        if patient_id in patient_files:
            data_file = os.path.join(data_dir, patient_files[patient_id])
            data = self._read_patient_file(data_file)
            data = torch.tensor([float(value) for value in data])
        else:  # Return all-zero tensor if data is missing
            eg_file = os.path.join(data_dir, list(patient_files.values())[0])
            nfeatures = len(self._read_patient_file(eg_file))
            data = torch.zeros(nfeatures)

        return data
    
    def _get_mRNA(self, data_dir, patient_id):
        return
    def _get_CNV(self, data_dir, patient_id):
        return
    
    def __getitem__(self,index):
        case_name = self.submitter_ids[index]
        
        data = {mod: self.data_loading_fns[mod](self.data_dirs[mod],case_name) for mod in self.modalities}
        data = self._data_to_device(data)
        
        label ={}
        label['time'] = self.label.loc[self.label['submitter_id'] == case_name, 'time'].values[0]
        label['time'] = torch.tensor(label['time'], dtype = float)
        label['event'] = self.label.loc[self.label['submitter_id'] == case_name, 'event'].values[0]
        label['event'] = torch.tensor(label['event'], dtype = float)
        label = self._data_to_device(label)
        
        return data, label, case_name
    
    def get(self,index):
        return self.__getitem__(index)
    
    def __len__(self):
        return len(self.submitter_ids)
    


class unified_dataset2(Dataset):
    def __init__(self, base_dir, label_path, modalities, device): # dir = ~/wsi/TCGA-KIRC/train or val or test
        super(unified_dataset2,self).__init__()
        
        self.modalities = modalities
        self.device = device
        
        self.base_dir = base_dir
        self.data_dirs = {x: os.path.join(base_dir, x) for x in modalities}
        self.label = pd.read_csv(label_path, sep='\t')
        group = base_dir.split('/')[-1]
        self.submitter_ids = list(self.label[self.label['group']==group]['submitter_id'])
        if 'gene' in modalities:
            sheets_dict = pd.read_excel('./TCGA-KIRC/TCGA-KIRC_(PanCancer)_고대전달본.xlsx', sheet_name=None)
            self.gene_df = sheets_dict['TCGA-KIRC_PC 356 Ps+ 11genes']
            self.gene_df.rename(columns={'Patient ID': 'submitter_id'}, inplace=True)
        self.data_loading_fns = {'clinical': self._get_clinical,
                                 'miRNA': self._get_miRNA,
                                #  'mRNA': self._get_mRNA,
                                #  'CNV': self._get_CNV,
                                 'ct': self._get_ct,
                                 'wsi': self._get_wsi,
                                 'gene': self._get_gene
                                 }
    
    def _data_to_device(self, data):
        for modality in data:
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple([v.to(self.device)
                                        for v in data[modality]])
            else:
                data[modality] = data[modality].to(self.device)

        return data

    def get_ct(self, data_dir, patient_id, folder):
        ct_data_dir = os.path.join(data_dir,folder)
        ct_case_dir = os.path.join(ct_data_dir, patient_id)
        if not os.path.exists(ct_case_dir):
            ct_data_list_ = torch.zeros((4,96,160,192), device=self.device)
        else:
            ct_data_list = [None, None,None,None] # N A P D
            
            case_file_lists = os.listdir(ct_case_dir)
            
            for file_name in case_file_lists:
                file = nib.load(os.path.join(ct_case_dir, file_name))
                data = file.get_fdata()
                mod = file_name.split('_')[1][0] # N A P D
                
                target_shape = (96, 160, 192)
                padding = []

                for i in range(3):
                    current_dim = data.shape[i] if i < len(data.shape) else 1
                    pad_size = max(0, target_shape[i] - current_dim)
                    pad_before = pad_size // 2
                    pad_after = pad_size - pad_before
                    padding.append((pad_before, pad_after))
                
                data = np.pad(data, padding, mode='constant')
                
                mod_index = {'N': 0, 'A': 1, 'P': 2, 'D': 3}.get(mod, None)
                if mod_index is not None:
                    # 텐서로 변환하고, 리스트에 저장
                    ct_data_list[mod_index] = torch.tensor(data, device=self.device, dtype=torch.float32).unsqueeze(0)
                    
            for i in range(4):
                if ct_data_list[i] is None:
                    # 이미 존재하는 데이터의 shape를 기준으로 0으로 채운 텐서 생성
                    dummy_shape = (1, 96, 160, 192) # (채널, 첫 번째 차원, 두 번째 차원, 세 번째 차원)
                    ct_data_list[i] = torch.zeros(dummy_shape, device=self.device, dtype=torch.float32)
            # ct_data_list_ = torch.stack(ct_data_list,dim=0)
            ct_data_list_ = torch.cat(ct_data_list,dim=0)
        
        return ct_data_list_
    
    def _get_ct(self, data_dir, patient_id):
        ct = self.get_ct(data_dir, patient_id, 'ct')
        ct_seg = self.get_ct(data_dir, patient_id, 'ct_seg')
        return (ct, ct_seg)
    
    def _get_clinical(self, data_dir, patient_id):
        data_file = os.path.join(data_dir, patient_id+'.tsv')
        data = self._read_patient_file(data_file)
        categorical = torch.tensor(
            [int(float(value)) for value in data[:4]], dtype=torch.int)
        continuous = torch.tensor(
            [float(value) for value in data[4:5]], dtype=torch.float)

        return categorical, continuous
    
    def _get_wsi(self, data_dirs, case_name):
        data_file_lists = os.listdir(data_dirs)
        file_names = [file for file in data_file_lists if file.startswith(case_name)]
        wsi_data_list = []
        
        for i, file_name in enumerate(file_names):
            file = np.load(os.path.join(data_dirs,file_name),allow_pickle=True)
            feature = file['features']
            edge_index = torch.tensor(file['edge_index'].astype(np.int64))
                
            data = Data(x=torch.Tensor(feature).to(self.device), edge_index=edge_index.to(self.device))
            wsi_data_list.append(data)
        return wsi_data_list
    
    def _read_patient_file(self, path):
        with open(path, 'r') as f:
            f = csv.reader(f, delimiter='\t')

            values = []
            for row in enumerate(f):
                values.append(row[1][0])

        return values
    
    def _get_miRNA(self, data_dir, patient_id):
        patient_files = {os.path.splitext(f)[0]: f
                         for f in os.listdir(data_dir)}

        if patient_id in patient_files:
            data_file = os.path.join(data_dir, patient_files[patient_id])
            data = self._read_patient_file(data_file)
            data = torch.tensor([float(value) for value in data])
        else:  # Return all-zero tensor if data is missing
            eg_file = os.path.join(data_dir, list(patient_files.values())[0])
            nfeatures = len(self._read_patient_file(eg_file))
            data = torch.zeros(nfeatures)

        return data
    
    def _get_gene(self, data_dir, patient_id):
        patient_files = list(self.gene_df['submitter_id'])
        cols = ['ACSS3', 'ALG13', 'ASXL3', 'BAP1', 'CFP', 'FAM47A', 'HAUS7', 'JADE3', 'KDM6A', 'NBPF10', 'NCOR1P1', 'SCRN1', 'ZNF449']
        if patient_id in patient_files:
            data = torch.tensor(self.gene_df[self.gene_df['submitter_id']==patient_id].loc[:,cols].values[0], dtype=torch.float32)
        else:
            nfeatures = len(cols)
            data = torch.zeros(nfeatures, dtype=torch.float32)
        return data
    
    def __getitem__(self,index):
        case_name = self.submitter_ids[index]
        
        data = {mod: self.data_loading_fns[mod](self.data_dirs[mod],case_name) for mod in self.modalities}
        data = self._data_to_device(data)
        
        label ={}
        label['time'] = self.label.loc[self.label['submitter_id'] == case_name, 'time'].values[0]
        label['time'] = torch.tensor(label['time'], dtype = float)
        label['event'] = self.label.loc[self.label['submitter_id'] == case_name, 'event'].values[0]
        label['event'] = torch.tensor(label['event'], dtype = float)
        label = self._data_to_device(label)
        
        return data, label, case_name
    
    def get(self,index):
        return self.__getitem__(index)
    
    def __len__(self):
        return len(self.submitter_ids)