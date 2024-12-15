
import sys
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from basicsr.models import create_model
from basicsr.utils.options import parse
from utils import my_summary


def model_sparsity(model):
    total = 0
    non_zero = 0
    total_features = 0
    non_zero_features = 0
    total_classifiers = 0
    non_zero_classifiers = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            total += module.weight.nelement()
            non_zero += module.weight.nonzero().size(0)
            if isinstance(module, torch.nn.Conv2d):
                total_features += module.weight.nelement()
                non_zero_features += module.weight.nonzero().size(0)
            elif isinstance(module, torch.nn.Linear):
                total_classifiers += module.weight.nelement()
                non_zero_classifiers += module.weight.nonzero().size(0)

    sparsity = (1 - non_zero / total) * 100
    sparsity_features = (1 - non_zero_features / total_features) * 100
    sparsity_classifiers = (1 - non_zero_classifiers / total_classifiers) * 100

    print(f"Overall Sparsity: {sparsity:.2f}%")
    print(f"Features Sparsity: {sparsity_features:.2f}%")
    print(f"Classifiers Sparsity: {sparsity_classifiers:.2f}%")


    return sparsity

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

model.to('cpu')


print('Before pruning')
my_summary(model, 256, 256, 3, 1)

'''
Prune the model

RetinexFormer(
  (body): Sequential(
    (0): RetinexFormer_Single_Stage(
      (estimator): Illumination_Estimator(
        (conv1): Conv2d(4, 40, kernel_size=(1, 1), stride=(1, 1))
        (depth_conv): Conv2d(40, 40, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=4)
        (conv2): Conv2d(40, 3, kernel_size=(1, 1), stride=(1, 1))
      )
      (denoiser): Denoiser(
        (embedding): Conv2d(3, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (encoder_layers): ModuleList(
          (0): ModuleList(
            (0): IGAB(
              (blocks): ModuleList(
                (0): ModuleList(
                  (0): IG_MSA(
                    (to_q): Linear(in_features=40, out_features=40, bias=False)
                    (to_k): Linear(in_features=40, out_features=40, bias=False)
                    (to_v): Linear(in_features=40, out_features=40, bias=False)
                    (proj): Linear(in_features=40, out_features=40, bias=True)
                    (pos_emb): Sequential(
                      (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
                      (1): GELU()
                      (2): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
                    )
                  )
                  (1): PreNorm(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Conv2d(40, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (1): GELU()
                        (2): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=160, bias=False)
                        (3): GELU()
                        (4): Conv2d(160, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      )
                    )
                    (norm): LayerNorm((40,), eps=1e-05, elementwise_affine=True)
                  )
                )
              )
            )
            (1): Conv2d(40, 80, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
            (2): Conv2d(40, 80, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          )
          (1): ModuleList(
            (0): IGAB(
              (blocks): ModuleList(
                (0): ModuleList(
                  (0): IG_MSA(
                    (to_q): Linear(in_features=80, out_features=80, bias=False)
                    (to_k): Linear(in_features=80, out_features=80, bias=False)
                    (to_v): Linear(in_features=80, out_features=80, bias=False)
                    (proj): Linear(in_features=80, out_features=80, bias=True)
                    (pos_emb): Sequential(
                      (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
                      (1): GELU()
                      (2): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
                    )
                  )
                  (1): PreNorm(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (1): GELU()
                        (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=320, bias=False)
                        (3): GELU()
                        (4): Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      )
                    )
                    (norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
                  )
                )
                (1): ModuleList(
                  (0): IG_MSA(
                    (to_q): Linear(in_features=80, out_features=80, bias=False)
                    (to_k): Linear(in_features=80, out_features=80, bias=False)
                    (to_v): Linear(in_features=80, out_features=80, bias=False)
                    (proj): Linear(in_features=80, out_features=80, bias=True)
                    (pos_emb): Sequential(
                      (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
                      (1): GELU()
                      (2): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
                    )
                  )
                  (1): PreNorm(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (1): GELU()
                        (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=320, bias=False)
                        (3): GELU()
                        (4): Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      )
                    )
                    (norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
                  )
                )
              )
            )
            (1): Conv2d(80, 160, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
            (2): Conv2d(80, 160, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
          )
        )
        (bottleneck): IGAB(
          (blocks): ModuleList(
            (0): ModuleList(
              (0): IG_MSA(
                (to_q): Linear(in_features=160, out_features=160, bias=False)
                (to_k): Linear(in_features=160, out_features=160, bias=False)
                (to_v): Linear(in_features=160, out_features=160, bias=False)
                (proj): Linear(in_features=160, out_features=160, bias=True)
                (pos_emb): Sequential(
                  (0): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=160, bias=False)
                  (1): GELU()
                  (2): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=160, bias=False)
                )
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (net): Sequential(
                    (0): Conv2d(160, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (1): GELU()
                    (2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640, bias=False)
                    (3): GELU()
                    (4): Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  )
                )
                (norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
              )
            )
            (1): ModuleList(
              (0): IG_MSA(
                (to_q): Linear(in_features=160, out_features=160, bias=False)
                (to_k): Linear(in_features=160, out_features=160, bias=False)
                (to_v): Linear(in_features=160, out_features=160, bias=False)
                (proj): Linear(in_features=160, out_features=160, bias=True)
                (pos_emb): Sequential(
                  (0): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=160, bias=False)
                  (1): GELU()
                  (2): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=160, bias=False)
                )
              )
              (1): PreNorm(
                (fn): FeedForward(
                  (net): Sequential(
                    (0): Conv2d(160, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)
                    (1): GELU()
                    (2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=640, bias=False)
                    (3): GELU()
                    (4): Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                  )
                )
                (norm): LayerNorm((160,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
        )
        (decoder_layers): ModuleList(
          (0): ModuleList(
            (0): ConvTranspose2d(160, 80, kernel_size=(2, 2), stride=(2, 2))
            (1): Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (2): IGAB(
              (blocks): ModuleList(
                (0): ModuleList(
                  (0): IG_MSA(
                    (to_q): Linear(in_features=80, out_features=80, bias=False)
                    (to_k): Linear(in_features=80, out_features=80, bias=False)
                    (to_v): Linear(in_features=80, out_features=80, bias=False)
                    (proj): Linear(in_features=80, out_features=80, bias=True)
                    (pos_emb): Sequential(
                      (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
                      (1): GELU()
                      (2): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
                    )
                  )
                  (1): PreNorm(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (1): GELU()
                        (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=320, bias=False)
                        (3): GELU()
                        (4): Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      )
                    )
                    (norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
                  )
                )
                (1): ModuleList(
                  (0): IG_MSA(
                    (to_q): Linear(in_features=80, out_features=80, bias=False)
                    (to_k): Linear(in_features=80, out_features=80, bias=False)
                    (to_v): Linear(in_features=80, out_features=80, bias=False)
                    (proj): Linear(in_features=80, out_features=80, bias=True)
                    (pos_emb): Sequential(
                      (0): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
                      (1): GELU()
                      (2): Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=80, bias=False)
                    )
                  )
                  (1): PreNorm(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (1): GELU()
                        (2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=320, bias=False)
                        (3): GELU()
                        (4): Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      )
                    )
                    (norm): LayerNorm((80,), eps=1e-05, elementwise_affine=True)
                  )
                )
              )
            )
          )
          (1): ModuleList(
            (0): ConvTranspose2d(80, 40, kernel_size=(2, 2), stride=(2, 2))
            (1): Conv2d(80, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (2): IGAB(
              (blocks): ModuleList(
                (0): ModuleList(
                  (0): IG_MSA(
                    (to_q): Linear(in_features=40, out_features=40, bias=False)
                    (to_k): Linear(in_features=40, out_features=40, bias=False)
                    (to_v): Linear(in_features=40, out_features=40, bias=False)
                    (proj): Linear(in_features=40, out_features=40, bias=True)
                    (pos_emb): Sequential(
                      (0): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
                      (1): GELU()
                      (2): Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=40, bias=False)
                    )
                  )
                  (1): PreNorm(
                    (fn): FeedForward(
                      (net): Sequential(
                        (0): Conv2d(40, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
                        (1): GELU()
                        (2): Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=160, bias=False)
                        (3): GELU()
                        (4): Conv2d(160, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
                      )
                    )
                    (norm): LayerNorm((40,), eps=1e-05, elementwise_affine=True)
                  )
                )
              )
            )
          )
        )
        (mapping): Conv2d(40, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (lrelu): LeakyReLU(negative_slope=0.1, inplace=True)
      )
    )
  )
)

'''

print('Model named modules')

for name, module in model.named_modules():
    print(name, module)


    # iteratively go inside the nested modules

    if module.__class__.__name__ == 'Illumination_Estimator':
        for name, submodule in module.named_modules():
            print(name, submodule)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'Denoiser':
        for name, submodule in module.named_modules():
            print(name, submodule)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'IGAB':
        for name, submodule in module.named_modules():
            print(name, submodule)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'IG_MSA':
        for name, submodule in module.named_modules():
            print(name, submodule)

            if submodule.__class__.__name__ == 'Linear':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'PreNorm':
        for name, submodule in module.named_modules():
            print(name, submodule)

            if submodule.__class__.__name__ == 'Linear':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'FeedForward':
        for name, submodule in module.named_modules():
            print(name, submodule)

            if submodule.__class__.__name__ == 'Linear':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'LayerNorm':
        for name, submodule in module.named_modules():
            print(name, submodule)
                
            if submodule.__class__.__name__ == 'Linear':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'ConvTranspose2d':
        for name, submodule in module.named_modules():
            print(name, submodule)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'Sequential':
        for name, submodule in module.named_modules():
            print(name, submodule)

            if submodule.__class__.__name__ == 'Conv2d':
                prune.l1_unstructured(submodule, name='weight', amount=0.5)

    if module.__class__.__name__ == 'Conv2d':
        prune.l1_unstructured(module, name='weight', amount=0.5)

    if module.__class__.__name__ == 'Linear':
        prune.l1_unstructured(module, name='weight', amount=0.5)

    if module.__class__.__name__ == 'ConvTranspose2d':
        prune.l1_unstructured(module, name='weight', amount=0.5)


        
print('After pruning')

model_sparsity(model)
my_summary(model, 256, 256, 3, 1)

'''
Run the following command to benchmark the model
python Enhancement/pruning_the_models.py pretrained_weights/LOL_v1.pth /home/ubuntu/work/ForkedRetinexformer/Options/RetinexFormer_LOL_v1.yml
python Enhancement/pruning_the_models.py pretrained_weights/FiveK.pth Options/RetinexFormer_FiveK.yml
'''







