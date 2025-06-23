"""Deep Learning-based multimodal data model for survival prediction."""

import warnings

import torch

from sub_models import FC, ClinicalNet, CnvNet, WsiNet, Fusion, wsi_model, BaseNet, ClinicalNet2, BaseNet_multi, miRNA, GeneNet


class MultiSurv(torch.nn.Module):
    """Deep Learning model for MULTImodal pan-cancer SURVival prediction."""
    def __init__(self, data_modalities, WSI=False, CT=False, fusion_method='cat',
                 n_output_intervals=None, device='cuda'):
        super(MultiSurv, self).__init__()
        self.data_modalities = data_modalities
        self.mfs = 256
        self.WSI = WSI
        self.CT = CT
        valid_mods = ['clinical', 'wsi', 'mRNA', 'miRNA', 'DNAm', 'CNV', 'ct']
        assert all(mod in valid_mods for mod in data_modalities), \
                f'Accepted input data modalitites are: {valid_mods}'

        assert len(data_modalities) > 0, 'At least one input must be provided.'

        self.num_features = 0
        if fusion_method == 'cat':
            if WSI:
                self.num_features += self.mfs 
            if CT:
                self.num_features += 64 
        else:
            self.num_features = self.mfs

        self.submodels = {}

        # Clinical -----------------------------------------------------------#
        if 'clinical' in self.data_modalities:
            self.clinical_submodel = ClinicalNet(
                output_vector_size=self.mfs)
            self.submodels['clinical'] = self.clinical_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # WSI patches --------------------------------------------------------#
        # if 'wsi' in self.data_modalities:
        #     self.wsi_submodel = GCN_Risk_Only(device=device)
        #     self.submodels['wsi'] = self.wsi_submodel
        #     if fusion_method == 'cat':
        #         self.num_features += self.mfs

        # mRNA ---------------------------------------------------------------#
        if 'mRNA' in self.data_modalities:
            self.mRNA_submodel = FC(1000, self.mfs, 3)
            self.submodels['mRNA'] = self.mRNA_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # miRNA --------------------------------------------------------------#
        if 'miRNA' in self.data_modalities:
            self.miRNA_submodel = FC(1881, self.mfs, 3, scaling_factor=2)
            self.submodels['miRNA'] = self.miRNA_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # DNAm ---------------------------------------------------------------#
        if 'DNAm' in self.data_modalities:
            self.DNAm_submodel = FC(5000, self.mfs, 5, scaling_factor=2)
            self.submodels['DNAm'] = self.DNAm_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # CNV ---------------------------------------------------------------#
        if 'CNV' in self.data_modalities:
            self.CNV_submodel = CnvNet(output_vector_size=self.mfs)
            self.submodels['CNV'] = self.CNV_submodel

            if fusion_method == 'cat':
                self.num_features += self.mfs

        # Instantiate multimodal aggregator ----------------------------------#
        if len(data_modalities) > 1:
            self.aggregator = Fusion(fusion_method, self.mfs, device)
        else:
            if fusion_method is not None:
                print("Input data is unimodal: no fusion procedure.")
                # warnings.warn('Input data is unimodal: no fusion procedure.')

        # Fully-connected and risk layers ------------------------------------#
        n_fc_layers = 4
        n_neurons = 128

        self.fc_block = FC(
            in_features=self.num_features,
            out_features=n_neurons,
            n_layers=n_fc_layers)

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons,
                            out_features=n_output_intervals),
            # torch.nn.Sigmoid()
        )

    def forward(self, x, wsi_feature, ct_feature):
        multimodal_features = tuple()

        if self.WSI:
            multimodal_features += (wsi_feature,)
            feature_repr = {'modalities': multimodal_features, 'fused': x}
        
        if self.CT:
            multimodal_features += (ct_feature,)
            feature_repr = {'modalities': multimodal_features, 'fused': x}
            
        # Run data through modality sub-models (generate feature vectors) ----#
        for modality in x:
            multimodal_features += (self.submodels[modality](x[modality]),)

        # Feature fusion/aggregation -----------------------------------------#
        if (len(multimodal_features) > 1):
            x1 = self.aggregator(torch.stack(multimodal_features))
            feature_repr = {'modalities': multimodal_features, 'fused': x}
        else:  # skip if running unimodal data
            x1 = multimodal_features[0]
            feature_repr = {'modalities': multimodal_features[0]}
            
        # Outputs ------------------------------------------------------------#
        x = self.fc_block(x1) # [B,256] -> [B,128]
        risk = self.risk_layer(x) # [B,128] -> [B,1]

        return risk, x1
    
class MultiSurv2(torch.nn.Module):
    def __init__(self, modalities, fusion_method='cat', device='cuda', finetune=False):
        super(MultiSurv2, self).__init__()
        
        self.modalities = modalities
        self.submodels = {}
        self.num_features = 0
        self.finetune = finetune
        
        # Clinical -----------------------------------------------------------#
        if 'clinical' in self.modalities:
            self.clinical_submodel = ClinicalNet2(output_vector_size=256)
            self.submodels['clinical'] = self.clinical_submodel

            if self.finetune:
                ckpt = './ckpt/best_model_clinical_0.688199.pth'
                ckeckpoint = torch.load(ckpt)
                filtered_state_dict = {key: value for key, value in ckeckpoint.items() if not (key.startswith('lin') or key.startswith('lin2'))}
                self.clinical_submodel.load_state_dict(filtered_state_dict, strict=False)
                for name, param in self.clinical_submodel.named_parameters():
                    if not name.startswith('lin'):  
                        param.requires_grad = False
            
            if fusion_method == 'cat':
                self.num_features += 256
                
        # WSI ----------------------------------------------------------------#
        if 'wsi' in self.modalities:
            self.wsi_submodel = wsi_model()
            
            if self.finetune:
                # ckpt = './ckpt/best_model_wsi_0.829_hj.pth'
                ckpt = './ckpt/best_model_wsi_0.829_hj.pth'
                ckeckpoint = torch.load(ckpt)
                filtered_state_dict = {key: value for key, value in ckeckpoint.items() if not (key.startswith('lin') or key.startswith('lin2'))}
                self.wsi_submodel.load_state_dict(filtered_state_dict, strict=False)
                for name, param in self.wsi_submodel.named_parameters():
                    if not name.startswith('lin'):  
                        param.requires_grad = False
            
            self.submodels['wsi'] = self.wsi_submodel
            
            if fusion_method == 'cat':
                self.num_features += 256
                
        # miRNA --------------------------------------------------------------#
        if 'miRNA' in self.modalities:
            self.miRNA_submodel = miRNA()
            # self.miRNA_submodel = FC(1881, self.num_features, 2)

            if self.finetune:
                ckpt = './ckpt/best_model_miRNA_0.669312.pth'
                ckeckpoint = torch.load(ckpt)
                filtered_state_dict = {key: value for key, value in ckeckpoint.items() if not (key.startswith('lin') or key.startswith('lin2'))}
                self.miRNA_submodel.load_state_dict(filtered_state_dict, strict=False)
                for name, param in self.miRNA_submodel.named_parameters():
                    if not name.startswith('lin'):  
                        param.requires_grad = False

            self.submodels['miRNA'] = self.miRNA_submodel

            if fusion_method == 'cat':
                self.num_features += 256
        
        # CT -----------------------------------------------------------------#
        if 'ct' in self.modalities:
            self.CT_submodel = BaseNet_multi()
            self.submodels['ct'] = self.CT_submodel

            if fusion_method == 'cat':
                self.num_features += 64
                
        # gene -----------------------------------------------------------------#
        if 'gene' in self.modalities:
            self.gene_submodel = GeneNet()
            self.submodels['gene'] = self.gene_submodel

            if fusion_method == 'cat':
                self.num_features += 64
                
        # Aggregater ---------------------------------------------------------#
        if len(modalities) > 1:
            self.aggregator = Fusion(fusion_method, self.num_features, device)
            
        # Fusion -------------------------------------------------------------#
        if len(modalities) > 1:
            n_fc_layers = 4
            n_neurons = 128

            self.fc_block = FC(
                in_features=self.num_features,
                out_features=n_neurons,
                n_layers=n_fc_layers)

            self.risk_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features=n_neurons,
                                out_features=1),
                # torch.nn.Sigmoid()
            )
        
    def forward(self, x):
        if (len(self.modalities) == 1):
            risk, feature = self.submodels[self.modalities[0]](x[self.modalities[0]])
            return risk
        
        
        else:
            multimodal_features = []
            for modality in x:
                multimodal_features.append(self.submodels[modality](x[modality])[1])
            # Aggregater ---------------------------------------------------------#
            x1 = self.aggregator(torch.cat(multimodal_features, dim=1))
            # x1 = self.aggregator(multimodal_features)
            
            # Fusion -------------------------------------------------------------#
            x = self.fc_block(x1) # [B,256] -> [B,128]
            risk = self.risk_layer(x) # [B,128] -> [B,1]

        return risk
