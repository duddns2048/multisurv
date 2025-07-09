"""Deep Learning-based multimodal data model for survival prediction."""

import warnings

import torch

from sub_models import FC, CnvNet, WsiNet, Fusion, wsi_model, BaseNet, ClinicalNet, BaseNet_multi, miRNA, GeneNet

    
class MultiSurv(torch.nn.Module):
    def __init__(self, modalities, fusion_method='cat', device='cuda', finetune=False):
        super(MultiSurv, self).__init__()
        
        self.modalities = modalities
        self.submodels = {}
        self.num_features = 0
        self.finetune = finetune
        
        # Clinical -----------------------------------------------------------#
        if 'clinical' in self.modalities:
            self.clinical_submodel = ClinicalNet(output_vector_size=256)
            self.submodels['clinical'] = self.clinical_submodel

            if self.finetune:
                ckpt = './ckpt/best_model_clinical_0.688199.pth'
                self.load_ckpt(self.clinical_submodel, ckpt)
            
            if fusion_method == 'cat':
                self.num_features += 256
                
        # WSI ----------------------------------------------------------------#
        if 'wsi' in self.modalities:
            self.wsi_submodel = wsi_model()
            
            if self.finetune:
                ckpt = './ckpt/best_model_wsi_0.829_hj.pth'
                self.load_ckpt(self.wsi_submodel, ckpt)
            
            self.submodels['wsi'] = self.wsi_submodel
            
            if fusion_method == 'cat':
                self.num_features += 256
                
        # miRNA --------------------------------------------------------------#
        if 'miRNA' in self.modalities:
            self.miRNA_submodel = miRNA()

            if self.finetune:
                ckpt = './ckpt/best_model_miRNA_0.669312.pth'
                self.load_ckpt(self.miRNA_submodel, ckpt)

            self.submodels['miRNA'] = self.miRNA_submodel

            if fusion_method == 'cat':
                self.num_features += 256
        
        # CT -----------------------------------------------------------------#
        if 'ct' in self.modalities:
            self.CT_submodel = BaseNet_multi()

            # if self.finetune:
            #     ckpt = './ckpt/best_model_miRNA_0.669312.pth'
            #     self.load_ckpt(self.miRNA_submodel, ckpt)

            self.submodels['ct'] = self.CT_submodel

            if fusion_method == 'cat':
                self.num_features += 256
                
        # gene -----------------------------------------------------------------#
        if 'gene' in self.modalities:
            self.gene_submodel = GeneNet()

            # if self.finetune:
            #     ckpt = './ckpt/best_model_miRNA_0.669312.pth'
            #     self.load_ckpt(self.miRNA_submodel, ckpt)

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
    def load_ckpt(self, model, ckpt):
        ckeckpoint = torch.load(ckpt)
        filtered_state_dict = {key: value for key, value in ckeckpoint.items() if not (key.startswith('lin') or key.startswith('lin2'))}
        model.load_state_dict(filtered_state_dict, strict=False)
        for name, param in model.named_parameters():
            if not name.startswith('lin'):  
                param.requires_grad = False
        return model
        


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
