import sys
import torch

from basicsr.models import create_model
from basicsr.utils.options import parse
from utils import my_summary

# load the model
if len(sys.argv) != 3:
    print("Usage: python benchmark_dl.py <model_file_path> <option-fle-path>")
    sys.exit(1)

model_file_path = sys.argv[1]
option_file_path = sys.argv[2]

def load_model(opt_path, weight_path):
    # Parse the configuration file
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False

    # Create and load model
    model = create_model(opt).net_g
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint["params"])
    model = model.cuda().eval()
    return model

model = load_model(option_file_path, model_file_path)

my_summary(model, 256, 256, 3, 1)

'''
Run the following command to benchmark the model
python Enhancement/benchmark_dl.py pretrained_weights/LOL_v1.pth /home/ubuntu/work/ForkedRetinexformer/Options/RetinexFormer_LOL_v1_benchmark.yml
python Enhancement/benchmark_dl.py pretrained_weights/FiveK.pth Options/RetinexFormer_FiveK.yml
'''
