'''
Download models .pth files  from Gdrive and put in pretrained_weights
'''
import os
import argparse
import gdown

def download_model(model_name, model_id, model_path):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    model_file = os.path.join(model_path, f'{model_name}.pth')
    if os.path.exists(model_file):
        print(f'Model already downloaded at {model_file}')
        return

    print(f'Downloading {model_name} model')
    url = f'https://drive.google.com/uc?id={model_id}'
    output = f'{model_path}/{model_name}.pth'
    gdown.download(url, output, quiet=False)
    print(f'{model_name} model downloaded at {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download model from Gdrive')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--model_id', type=str, help='Gdrive id of the model')
    parser.add_argument('--model_path', type=str, help='Path to save the model')
    args = parser.parse_args()

    download_model(args.model_name, args.model_id, args.model_path)

'''
Lets run the script to download the model here https://drive.google.com/file/d/1oxvPPfhbOwZURTFenWnFp3H3Lakkqw3t/view

python basicsr/utils/download_models.py --model_name 'FiveK' --model_id '1oxvPPfhbOwZURTFenWnFp3H3Lakkqw3t' --model_path 'pretrained_weights'
python basicsr/utils/download_models.py --model_name 'LOL_v1' --model_id '1xDwQtTCj3tlAVCTJgYrzonBGVwqeOhKu' --model_path 'pretrained_weights'



'''
