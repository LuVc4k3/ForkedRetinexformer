'''
Downloads dataset from Gdrive if not already downloaded
'''

import os
import zipfile
import argparse
import gdown

def extract_zip(zip_path, target_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)

def download_dataset(dataset_name, dataset_id, dataset_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    elif os.path.exists(f'{dataset_path}/{dataset_name}'):
        print(f'Dataset already downloaded at {dataset_path}')
        return

    print(f'Downloading {dataset_name} dataset')
    url = f'https://drive.google.com/uc?id={dataset_id}'
    output = f'{dataset_path}/{dataset_name}.zip'
    gdown.download(url, output, quiet=False)
    extract_zip(output, dataset_path)
    os.remove(output)
    print(f'{dataset_name} dataset downloaded at {dataset_path}/{dataset_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dataset from Gdrive')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--dataset_id', type=str, help='Gdrive id of the dataset')
    parser.add_argument('--dataset_path', type=str, help='Path to save the dataset')
    args = parser.parse_args()

    download_dataset(args.dataset_name, args.dataset_id, args.dataset_path)


'''
Lets run the script to download the dataset here https://drive.google.com/file/d/11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR/view

python basicsr/utils/download_dataset.py --dataset_name 'FiveK' --dataset_id '11HEUmchFXyepI4v3dhjnDnmhW_DgwfRR' --dataset_path 'data'
python basicsr/utils/download_dataset.py --dataset_name 'LOL-v1' --dataset_id '1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H' --dataset_path 'data'

'''

