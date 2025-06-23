import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt

# from lifelines import KaplanMeierFitter
def save_value_counts_graphs(df):
    # 데이터프레임의 모든 컬럼에 대해 반복
    for column in df.columns:
        # 각 컬럼의 값에 대한 value_counts 계산
        value_counts = df[column].value_counts()

        # 막대그래프 그리기
        ax = value_counts.plot(kind='bar', figsize=(8, 6))

        # 그래프 제목 설정
        title = f'Clinical537 {column}'
        plt.title(title)

        # 그래프 파일로 저장 (파일명은 제목과 동일하게)
        plt.savefig(f'{title}.png')

        # 그래프 화면에 출력 후 닫기
        # plt.show()
        # plt.close()
        
DATA_LOCATION = './etc/raw_clinical.tsv'

clinical = pd.read_csv(DATA_LOCATION,sep='\t', na_values=["'--",'not reported', 'Not Reported'])

clinical = clinical.rename(columns={'case_submitter_id': 'submitter_id'})


# 전처리1 : 특정 열의 모든값이 다 다르면 삭제 (submitter_id 제외) & 특정 열이 모두 결측치면 삭제
n = clinical.shape[0] # shape (1074,213)
for col in clinical.columns:
    if col == 'submitter_id':
        continue
    n_levels = len(clinical[col].value_counts()) # nunique
    if n_levels == n: 
        clinical = clinical.drop(columns=[col]) # 특정 열의 모든 값이 다 다르면 삭제
    else:
        n_missing = sum(clinical[col].isnull())
        if n_missing > 0:
            if n_missing == n:
                clinical = clinical.drop(columns=[col]) # 특정 열이 모두 결측치 이면 삭제
            else:
                print(f'{col}: {n_missing} ({round(n_missing / n * 100, 2)}%)')

    # shape (1074,32)
# 전처리2: case_submitter_id 중복되는것들중에 데이터 가장 많은것만 남기고 삭제               
clinical['missing_count'] = clinical.isna().sum(axis=1)
clinical_sorted = clinical.sort_values(by='missing_count')
clinical = clinical_sorted.drop_duplicates(subset='submitter_id', keep='first').drop(columns='missing_count')
    # shape (537,32)
    
# 전처리3: TCIA pair 207 case
# file_path = './TCIA_KIRC_ids.txt' 
# with open(file_path, 'r') as file:
#     TCIA_paired_id = [line.strip() for line in file.readlines()]
# clinical = clinical[clinical['submitter_id'].isin(TCIA_paired_id)]
    # shape (207, 32)
    
# 전처리4: 사용할 column 고르기
label_cols = ['submitter_id', 'days_to_last_follow_up', 'vital_status', 'days_to_death']

# 'synchronous_malignancy' 빼: 하나빼고 전부 1임
# disease, treatments_pharmaceutical_treatment_or_therapy, treatments_radiation_treatment_or_therapy없어
keep_cols = [ 'age_at_diagnosis', 'prior_treatment', 'prior_malignancy',
              'gender', 'race', 'ethnicity', 'synchronous_malignancy']
columns_to_drop = [col for col in clinical.columns if col not in label_cols + keep_cols]
clinical = clinical.drop(columns=columns_to_drop)

clinical = clinical.set_index('submitter_id')
    # shape (207,11)

# 전처리5: race & ethnicity: race가 white이거나 비었으면 ethnicity값으로 채우고 ethnicity 제거
race_subset = (clinical['race'] == 'white')
ethnicity_subset = (~clinical['ethnicity'].isnull() &
                    (clinical['ethnicity'] == 'hispanic or latino'))
subset = race_subset & ethnicity_subset
clinical.loc[subset, 'race'] = clinical.loc[subset, 'ethnicity']

clinical = clinical.drop('ethnicity', axis=1)

# 전처리6: Vital_status와 추적날짜 일치시키기 
## Alive = days_to_last_follow_up
## Dead = days_to_death
subset = clinical.vital_status == 'Dead'
clinical.loc[subset, 'days_to_last_follow_up'] = None

# 전처리7: Label data

def get_duration(vital_status, days_to_death, days_to_last_follow_up):
    if vital_status == 'Dead':
        return days_to_death
    elif vital_status == 'Alive':
        return days_to_last_follow_up
    else:
        print('Found NaN in duration!')

def get_events(vital_status):
    if vital_status in ['1', 'Dead']:
        return 1
    elif vital_status in ['0', 'Alive']:
        return 0
    else:
        print('Found NaN in vital status!')
        
d = {'submitter_id': clinical.index,
     'time': clinical.apply(
         lambda x: get_duration(x['vital_status'], x['days_to_death'],
                                x['days_to_last_follow_up']), axis=1).values,
     'event': clinical.apply(
         lambda x: get_events(x['vital_status']), axis=1).values}
survival = pd.DataFrame.from_dict(d).astype(
    {'submitter_id': 'object', 'time': 'int64', 'event': 'int64'}) 

# 전처리8: 인덱스 col지정
clinical = clinical.join(survival.set_index('submitter_id'))       
    # shape (207,11)
# 전처리9: Train / Val / Test split
X = clinical
y = clinical[['time']]

# X_train, X_val, _, _ = train_test_split(
#     X, y, test_size=0.1, random_state=42, stratify=clinical[['event']])

# X = X_train
# y = X_train[['time']]

# import pandas as pd 
# label207 = pd.read_csv('./processed/labels_train&test.tsv',sep='\t', index_col=0)
# label207_ids = list(label207['submitter_id'])
# X = clinical[~clinical.index.isin(label207_ids)]
# y = X[['time']]

X_train, X_test, _, _ = train_test_split(
    X, y, test_size=0.1878, random_state=42, stratify=X[['event']])

def get_split_group(id_code):
    if id_code in list(X_train.index):
        return 'train'
    # elif id_code in X_val.index:
    #     return 'val'
    elif id_code in X_test.index:
        return 'test'
    else:
        print('Found NaN!')
        
clinical['group'] = 'Missing'
clinical['group'] = [get_split_group(x) for x in list(clinical.index)]
# for i in label207_ids:
#     clinical.loc[i,'group']=label207.loc[label207['submitter_id']==i,'group'].values[0]

# 전처리10: 연도, 일수 단위 바꾸기
# clinical['time'] = clinical['time'] / 365

'''
def get_data_group(df, value='train'):
    group = df.loc[df['group'] == value]
    return group.drop(columns='group')

train = get_data_group(clinical, 'train')
val = get_data_group(clinical, 'val')
test = get_data_group(clinical, 'test')
'''

# 전처리 11: 비어있는 col 최빈값, 평균값으로 채우기. input missing values 
def input_missing_values(feature, df):
    train_subset = df.loc[df['group'] == 'train', feature]
    try:
        input_value = train_subset.median()
        print(f'Median "{feature}": {input_value}')
    except TypeError:
        input_value = train_subset.mode().iloc[0]
        print(f'Mode "{feature}": {input_value}')

    df[feature].fillna(input_value, inplace=True)
    
    return df

# clinical = input_missing_values(feature='synchronous_malignancy', df=clinical)
# clinical = clinical.drop('synchronous_malignancy')


# 전처리12: Continuous Variable Scaling
id_groups = {
    'train': list(clinical.loc[clinical['group'] == 'train', ].index),
    'val': list(clinical.loc[clinical['group'] == 'val', ].index),
    'test': list(clinical.loc[clinical['group'] == 'test', ].index)}

continuous = ['age_at_diagnosis']

def min_max_scale(data, features, groups):
    train = data.loc[data.index.isin(groups['train']), features]

    scaler = MinMaxScaler()
    columns = train.columns
    scaler = scaler.fit(train[columns])
    
    data.loc[data.index.isin(groups['train']), features] = scaler.transform(
        train)
    # data.loc[data.index.isin(groups['val']), features] = scaler.transform(
    #     data.loc[data.index.isin(groups['val']), features])
    data.loc[data.index.isin(groups['test']), features] = scaler.transform(
        data.loc[data.index.isin(groups['test']), features])
    
    return data

clinical = min_max_scale(data=clinical, features=continuous, groups=id_groups)

# 전처리 13: Encode Categorical Variables
clinical = clinical.drop(columns=['days_to_death', 'days_to_last_follow_up', 'vital_status'])
skip = ['time', 'event', 'group']
categorical = [col for col in clinical.columns if col not in skip + continuous]
clinical = clinical[categorical + continuous + skip]

split_groups = ['train', 'val', 'test']

label_encoders = {}
for feature in categorical:
    clinical[feature] = clinical[feature].astype(str)
    label_encoders[feature] = LabelEncoder()
    label_encoders[feature].fit(
        clinical.loc[clinical['group'] == 'train', feature])

for group in split_groups:
    for feature in categorical:
        clinical.loc[
            clinical['group'] == group, feature] = label_encoders[feature].transform(
            clinical.loc[clinical['group'] == group, feature])
            

# Dimension embedding: 모델 embedding할때 필요
categorical_dims = [int(clinical[col].nunique()) for col in categorical]
embedding_dims = [(x, min(50, (x + 1) // 2)) for x in categorical_dims]

embedding_dims

print('Feature                                            Levels   Embedding dims')
print('-------                                            ------   --------------')
for i, feat in enumerate(categorical):
    print(feat, ' ' * (50 - len(feat)), embedding_dims[i][0],
          ' ' * (7 - len(str(embedding_dims[i][0]))), embedding_dims[i][1])
    
# Save data to files
label_columns = ['time', 'event', 'group']
survival = clinical.loc[:, label_columns]
# survival.to_csv('./TCGA-KIRC/clinical_537/labels.tsv',sep='\t', index=True)

def table_to_patient_files(table, dir_path, round_digits=4):
    n = len(table)
    
    i = 0

    for index, row in table.iterrows():
        print('\r' + f'Save data to files: {str(i + 1)}/{n}', end='')
        i+= 1

        target_file = os.path.join(dir_path, str(index) + '.tsv')
        
        with open(target_file, 'w') as f:
            if round_digits is not None:
                f.write('\n'.join(str(round(value, round_digits)) for value in row.values))
            else:
                f.write('\n'.join(str(value) for value in row.values))

    print()
    print()
clinical = clinical.drop(columns = 'group')
# table_to_patient_files(clinical, dir_path=DATA_LOCATION, round_digits=None)