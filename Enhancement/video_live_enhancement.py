'''
2. Enhance the video using the model.
3. Save the enhanced video feed in output dir .
'''

import cv2
import numpy as np
import os
import time
import torch
from PIL import Image
from torchvision import transforms
from basicsr.utils.options import parse
from basicsr.models import create_model

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

model = load_model('../Options/RetinexFormer_LOL_v1.yml', '../pretrained_weights/LOL_v1.pth')
model.eval()
model = model.cuda()

model.to(torch.device('cpu'))

torch.save(model, 'lolv1_model.pth')

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define the input and output directory
input_dir = 'input'
output_dir = 'output'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create the input directory if it does not exist
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

# load a video video.mp4
cap = cv2.VideoCapture('video.mp4')

# save 5fps frames to input directory
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % 12 == 0:
        cv2.imwrite(f'{input_dir}/frame_{frame_count}.png', frame)
    frame_count += 1

# Enhance the frames first convert each rgb to bgr
for frame in os.listdir(input_dir):
    frame_path = os.path.join(input_dir, frame)
    img = Image.open(frame_path).convert('RGB')
    img = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        enhanced_img = model(img)
    enhanced_img = enhanced_img.squeeze(0).cpu().detach().numpy()
    enhanced_img = np.clip(enhanced_img * 255, 0, 255).astype(np.uint8)
    enhanced_img = np.transpose(enhanced_img, (1, 2, 0))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, frame), enhanced_img)

# Release the video capture
cap.release()
cv2.destroyAllWindows()

# Create a video from the enhanced frames
img_array = []
for frame in os.listdir(output_dir):
    frame_path = os.path.join(output_dir, frame)
    img = cv2.imread(frame_path)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()

print('Enhanced video saved as output_video.mp4')



