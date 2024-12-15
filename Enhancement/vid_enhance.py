import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_model(opt_path, weight_path):
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False

    # Create and load model
    model = create_model(opt).net_g
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint["params"])
    model = model.cuda().eval()
    return model


def enhance_frame(model, frame):
    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame = (
        torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).cuda()
    )

    with torch.no_grad():
        enhanced = model(frame)

    enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)


def process_video_file(model, input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    with tqdm(total=total_frames, desc="Enhancing Video", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            enhanced_frame = enhance_frame(model, frame)
            out.write(enhanced_frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Enhanced video saved to: {output_video}")


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Video Enhancement using Retinexformer"
    )
    parser.add_argument(
        "--opt", required=True, type=str, help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to pretrained weights (.pth file)",
    )
    parser.add_argument(
        "--input_video", required=True, type=str, help="Path to input video file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save the enhanced video",
    )

    args = parser.parse_args()

    # Load the model
    model = load_model(args.opt, args.weights)
    print("Model loaded successfully.")

    # Prepare output path
    os.makedirs(args.output_dir, exist_ok=True)
    output_video = os.path.join(args.output_dir, "enhanced_video.mp4")

    # Process the video
    process_video_file(model, args.input_video, output_video)


if __name__ == "__main__":
    main()
