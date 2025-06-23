import os
import glob
import json
import math
import torch
import numpy as np
import cv2
import tqdm
from pathlib import Path
# from openslide import OpenSlide
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
# from histocartography.preprocessing import MacenkoStainNormalizer
# from TCGA_WSI.classifier_model import resnet_biclassifier


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def return_survival_time(slide, file_name):
    with open(file_name, 'r') as f:
        patients = json.load(f)
        # name = slide.split('-')[0]+'-'+slide.split('-')[1]+'-'+slide.split('-')[2]

        for patient in patients:
            patient_name = patient['diagnoses'][0]['submitter_id'].split('_')[0]
            if patient_name == name:
                try:
                    return patient['demographic']['days_to_death']
                except KeyError:
                    return -1
    return np.zeros(1)


def return_last_follow_up(slide, file_name):
    with open(file_name, 'r') as f:
        patients = json.load(f)
        name = slide.split('-')[0]+'-'+slide.split('-')[1]+'-'+slide.split('-')[2]

        for patient in patients:
            patient_name = patient['diagnoses'][0]['submitter_id'].split('_')[0]
            if patient_name == name:
                try:
                    return patient['diagnoses'][0]['days_to_last_follow_up']
                except KeyError:
                    return -1
    return np.zeros(1)


def return_patient_id(slide, file_name):
    with open(file_name, 'r') as f:
        patients = json.load(f)
        name = slide.split('-')[0]+'-'+slide.split('-')[1]+'-'+slide.split('-')[2]

        for patient in patients:
            patient_name = patient['diagnoses'][0]['submitter_id'].split('_')[0]
            if patient_name == name:
                return [name]
    return []


def return_status(survival_time):
    return 1 if survival_time != -1 else 0

def return_survive_in_5_years(survival_time, last_follow_up):
    if survival_time > 365 * 5 or last_follow_up > 365 * 5:
        return 1
    else:
        return 0

def get_features(slide, file_name, patches_dir, mag, coords):
    resnet18 = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(resnet18.children())[:-1])).cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feature_map = torch.tensor([])

    with open(file_name, 'r') as f:
        patients = json.load(f)
        name = slide.split('-')[0]+'-'+slide.split('-')[1]+'-'+slide.split('-')[2]

        for patient in patients:
            patient_name = patient['diagnoses'][0]['submitter_id'].split('_')[0]

            if patient_name == name:
                for patch in tqdm.tqdm(patches_dir):
                    img = cv2.imread(patch)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = color_normalizer.process(img)
                    img = transform(img).unsqueeze(0).cuda()

                    with torch.no_grad():
                        feature = model(img).squeeze(-1).squeeze(-1).cpu()
                        feature_map = torch.cat((feature_map, feature), 0)

                edge_index = get_adjacency(feature_map, coords, mag)
                break

    return feature_map, edge_index


def return_patch_coords(slide, file_name, patches_dir):
    pos_list = np.zeros((len(patches_dir), 2))

    with open(file_name, 'r') as f:
        patients = json.load(f)
        name = slide.split('-')[0]+'-'+slide.split('-')[1]+'-'+slide.split('-')[2]

        for patient in patients:
            patient_name = patient['diagnoses'][0]['submitter_id'].split('_')[0]
            if patient_name == name:
                for ii, patch in enumerate(patches_dir):
                    patch_name = Path(patch).stem
                    c_pos, r_pos = patch_name.split('_')[1], patch_name.split('_')[2]
                    pos_list[ii] = [int(c_pos), int(r_pos)]
                break

    return pos_list


def get_img_path(slide, file_name, patches_dir):
    path_list = []

    with open(file_name, 'r') as f:
        patients = json.load(f)
        name = slide.split('-')[0]+'-'+slide.split('-')[1]+'-'+slide.split('-')[2]

        for patient in patients:
            patient_name = patient['diagnoses'][0]['submitter_id'].split('_')[0]
            if patient_name == name:
                path_list = patches_dir
                break

    return path_list


def get_sorted_patch_coords(patch_list):
    coords = []
    idx = []

    for patch in patch_list:
        xx = int(Path(patch).stem.split('_')[1])
        yy = int(Path(patch).stem.split('_')[2])
        coords.append((xx, yy))

    for coord in coords:
        idx.append(math.sqrt(coord[0] + math.pow(coord[1], 2)))

    sorted_coords = np.array(coords)[np.argsort(idx)]
    return sorted_coords


def get_adjacency(points, coords, mag):
    n_p = points.shape[0]
    k_counter = 8
    
    if n_p < k_counter:
        k_counter = n_p
    
    edge_list_1 = []
    edge_list_2 = []

    if mag == '20X':
        dis_cut = 512
    elif mag == '5X':
        dis_cut = 128 * 5
    elif mag == '125X':
        dis_cut = 32 * 5

    for i in range(n_p):
        dist_list = [np.linalg.norm(coords[i] - coords[j]) for j in range(n_p)]
        sorted_list = np.argsort(dist_list)

        for k in range(k_counter):
            if dist_list[sorted_list[k]] <= dis_cut:
                edge_list_1.append(i)
                edge_list_2.append(sorted_list[k])

    edge_list = np.vstack([np.array(edge_list_1), np.array(edge_list_2)])
    return edge_list.astype(np.int32)


def Train_PCA_From_NPZ(npz_dir, out_dim=128):
    sel_files = []
    sel_idd = []

    npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))
    for cur_file in npz_files:
        print(f"Processing {cur_file} for PCA training")
        Train_file = np.load(cur_file)
        cur_features = Train_file['features']
        img_id = np.zeros(len(cur_features))

        all_fea = np.vstack(cur_features)
        img_id_batch = np.vstack(img_id)

        group_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1)
        for _, test_index in group_sss.split(all_fea, img_id_batch):
            sel_files.extend(all_fea[test_index])
            sel_idd.extend(img_id[test_index])

    train_pca_fea = np.vstack(sel_files)
    pca = PCA(n_components=out_dim, whiten=True).fit(train_pca_fea)
    return pca


def extract_fea_TCGA(npz_dir, pca_npz_dir, pca):
    npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))

    for cur_file in npz_files:
        file_name = os.path.basename(cur_file)
        print(f"Processing {file_name} for PCA transformation")
        Train_file = np.load(cur_file)

        fea_pca = pca.transform(Train_file['features'])

        save_dir = pca_npz_dir
        make_dirs(save_dir)
        save_path = os.path.join(save_dir, file_name)

        np.savez(save_path,
                 features=fea_pca,
                 patient_id=Train_file['patient_id'],
                 survival_time=Train_file['survival_time'],
                 status=Train_file['status'],
                 edge_index=Train_file['edge_index'])


if __name__ == '__main__':
    # color_normalizer = MacenkoStainNormalizer()
    dataset = ['KIRC']
    dataset_idx = 0

    TCGA_dir = './'
    patches_root_dir = '/ssd/Datasets/TCGA-KIRC/WSI/Patch/20x_512'
    mag = '20X'

    file_name = os.path.join(TCGA_dir,'clinical.json')
    npz_dir = os.path.join(TCGA_dir, 'KIRC_npz')

    # slides = os.listdir(patches_root_dir)
    # slides = sorted([slide for slide in slides if 'TCGA' in slide])
    slides=[0,1,2,3,4,5,6,7,8,9]
    # read txt file
    tumor_patch_list = []
    # with open('tumor_patches.txt', 'r') as file:
    #     for line in file:
    #         tumor_patch_list.append(line.strip())
    
    for slide in slides:
        # save_path = os.path.join(npz_dir, f'{slide}.npz')
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # if os.path.exists(save_path):
        #     print(f'{slide} exists')
        #     continue
        # if slide != 'TCGA-BP-4975-01A-01-BS1':
        #     continue
            

        print(f"Creating NPZ for slide {slide}, {dataset[dataset_idx]}, {mag}")

        survival_time = return_survival_time(slide, file_name)
        last_follow_up = return_last_follow_up(slide, file_name)
        patient_id = return_patient_id(slide, file_name)
        status = return_status(survival_time)
        survive_in_5_years = return_survive_in_5_years(survival_time, last_follow_up)

        patches_dir = [tumor_patch for tumor_patch in tumor_patch_list if slide in tumor_patch]
        if len(patches_dir) == 0:
            print(f'No tumor patches found for slide {slide}')
            continue
        sorted_coords = get_sorted_patch_coords(patches_dir)
        sorted_patches_dir = [os.path.join(patches_root_dir, slide, f"{slide}_{i[0]}_{i[1]}_512.png") for i in sorted_coords]

        coords = return_patch_coords(slide, file_name, sorted_patches_dir)
        features, edge_index = get_features(slide, file_name, sorted_patches_dir, mag, coords)
        img_path = get_img_path(slide, file_name, sorted_patches_dir)


        np.savez(save_path, 
                 patient_id=patient_id, 
                 survival_time=survival_time, 
                 status=status, 
                 survive_in_5_years=survive_in_5_years, 
                 edge_index=edge_index, 
                 features=features, 
                 img_path=img_path,
                 coords=coords, 
                 patient_full_id=slide, 
                 last_follow_up=last_follow_up)