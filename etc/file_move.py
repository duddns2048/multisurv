import os
import shutil
import pandas as pd
# 소스 폴더와 목적지 폴더 경로 설정
source_folder = './processed/clinical'  # 파일들이 담겨 있는 폴더 경로
destination_folder = './TCGA-KIRC/KIRC_npz/train/clinical'  # 파일을 옮길 목적지 폴더 경로


label = pd.read_csv('./processed/labels.tsv', sep ='\t')

# ID 목록을 저장한 파일 경로 설정
id_file_path = './TCIA_KIRC_ids.txt'

# ID 목록을 읽어오는 부분
with open(id_file_path, 'r') as f:
    ids = f.read().splitlines()  # 각 라인을 읽어서 리스트로 만듦

# 각 ID에 해당하는 파일들을 이동
for file_id in ids:
    file_name = f"{file_id}.tsv"  # 파일 이름 구성
    source_file_path = os.path.join(source_folder, file_name)  # 소스 파일의 전체 경로
    destination_file_path = os.path.join(destination_folder, file_name)  # 목적지 파일의 전체 경로

    # 파일이 소스 폴더에 존재하는지 확인
    if os.path.exists(source_file_path):
        shutil.copy(source_file_path, destination_file_path)  # 파일을 목적지로 이동
        print(f"Moved: {source_file_path} to {destination_file_path}")
    else:
        print(f"File not found: {source_file_path}")