import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from basicsr.models import create_model
from basicsr.utils.options import parse
import argparse


def load_frames_from_directory(directory, target_shape=(512, 960)):
    """
    Load frames from a directory, supporting .npy and .png files.
    Automatically resizes frames to the target shape if necessary.

    Args:
        directory (str): Directory containing the frame files.
        target_shape (tuple): Desired frame shape (height, width).

    Returns:
        list: List of frames with the target shape.
    """
    frame_files = sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith((".npy", ".png"))
        ]
    )

    if not frame_files:
        raise ValueError(f"No .npy or .png files found in the directory: {directory}")

    frames = []
    for frame_file in tqdm(frame_files, desc="Loading Frames"):
        if frame_file.endswith(".npy"):
            frame = np.load(frame_file)  # Load .npy file
        elif frame_file.endswith(".png"):
            frame = cv2.cvtColor(
                cv2.imread(frame_file), cv2.COLOR_BGR2RGB
            )  # Load .png file as RGB
        else:
            raise ValueError(f"Unsupported file format: {frame_file}")

        # Resize the frame if it does not match the target shape
        if frame.shape[:2] != target_shape:
            frame = cv2.resize(
                frame,
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        frames.append(frame)

    return frames


def save_frames_to_video(frames, output_video, fps=30):
    """
    Save frames as a .mp4 video.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    with tqdm(total=len(frames), desc=f"Writing Video: {output_video}") as pbar:
        for frame in frames:
            # Convert RGB (from loaded frames) to BGR (for OpenCV)
            bgr_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)
            pbar.update(1)

    video_writer.release()
    print(f"Video saved to: {output_video}")


def enhance_frames(model, frames):
    """
    Enhance frames using the Retinexformer model.
    """
    enhanced_frames = []
    for frame in tqdm(frames, desc="Enhancing Frames"):
        # Normalize and convert frame to tensor
        frame_tensor = (
            torch.tensor(frame / 255.0, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .cuda()
        )

        with torch.no_grad():
            enhanced_tensor = model(frame_tensor).clamp(0, 1)

        # Convert tensor back to numpy array
        enhanced_frame = (
            enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        )
        enhanced_frames.append(enhanced_frame.astype(np.uint8))

    return enhanced_frames


def load_retinexformer_model(opt_path, weight_path):
    """
    Load the Retinexformer model with given options and weights.
    """
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False
    model = create_model(opt).net_g
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint["params"])
    model = model.cuda().eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Process frames and create input and enhanced videos using Retinexformer"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Directory containing frames (.npy or .png)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save the output videos",
    )
    parser.add_argument(
        "--opt",
        required=True,
        type=str,
        help="Path to Retinexformer YAML configuration file",
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to Retinexformer pretrained weights (.pth)",
    )
    parser.add_argument(
        "--fps", default=30, type=int, help="Frames per second for the output videos"
    )
    args = parser.parse_args()

    # Load frames
    frames = load_frames_from_directory(args.input_dir)

    # Save input frames to a video
    os.makedirs(args.output_dir, exist_ok=True)
    input_video_path = os.path.join(args.output_dir, "input_video.mp4")
    save_frames_to_video(frames, input_video_path, fps=args.fps)

    # Load Retinexformer model
    model = load_retinexformer_model(args.opt, args.weights)

    # Enhance frames
    enhanced_frames = enhance_frames(model, frames)

    # Save enhanced frames to a video
    enhanced_video_path = os.path.join(args.output_dir, "enhanced_video.mp4")
    save_frames_to_video(enhanced_frames, enhanced_video_path, fps=args.fps)


if __name__ == "__main__":
    main()
