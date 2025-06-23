import os
import shutil
import pandas as pd

modality = 'miRNA'
sourc_dir = f'./TCGA-KIRC/clinical_537/whole'
# sourc_dir = f'./TCGA-KIRC/KIRC_npz/train/{modality}'
target_dir = './TCGA-KIRC/clinical_537/'
label = pd.read_csv('./TCGA-KIRC/clinical_537/labels.tsv', sep='\t')

label_dict = {id: split for id , split in zip(list(label['submitter_id']),list(label['group']))}
file_list = os.listdir(sourc_dir)

# for id in label_dict:
#     group = label_dict[id]
#     if str(group) == 'train':
#         continue
#     elif str(group) == 'val':
#         source_file = os.path.join(sourc_dir, id+'.tsv')
#         destination_dir = os.path.join(target_dir, str(group),modality,id+'.tsv')
#         if os.path.exists(source_file):
#             shutil.move(source_file,destination_dir)
#     elif str(group) == 'test':
#         source_file = os.path.join(sourc_dir, id+'.tsv')
#         destination_dir = os.path.join(target_dir, str(group),modality,id+'.tsv')
#         if os.path.exists(source_file):
#             shutil.move(source_file,destination_dir)



for id in label_dict:
    group = label_dict[id]
    files = [x for x in file_list if x.startswith(id)]
    for file in files:
        src_file = os.path.join(sourc_dir,file)
        destination_file = os.path.join(target_dir, str(group), file)
        if os.path.exists(src_file):
            shutil.copy(src_file,destination_file)
        
        