import os
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from torchvision import transforms


def load_and_process_frames(frame_folder, size=(224, 224)):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )

    frames = []
    for frame_name in sorted(os.listdir(frame_folder)):
        if frame_name.endswith(".png"):
            frame_path = os.path.join(frame_folder, frame_name)
            image = Image.open(frame_path).convert("RGB")
            tensor = transform(image)
            frames.append(tensor)

    return torch.stack(frames).permute(1, 0, 2, 3)


def save_frames_to_pt(output_folder, pt_file):
    all_tensors = []
    video_names = []

    for video_name in sorted(os.listdir(output_folder)):
        video_folder = os.path.join(output_folder, video_name)
        if os.path.isdir(video_folder):
            frames = load_and_process_frames(video_folder)
            all_tensors.append(frames)
            video_names.append(video_name)

    # Stack all tensors along a new dimension
    all_tensors = torch.stack(all_tensors)

    # Save the tensor and video names directly to a .pt file
    save_dict = {"frames": all_tensors, "video_names": video_names}
    torch.save(save_dict, pt_file)

    print(f"Saved all frames to {pt_file}")


output_folder = "/root/autodl-tmp/mmsd/mmsd_raw_data/utterances_final/frames"
parquet_file = (
    "/root/autodl-tmp/mmsd/mmsd_raw_data/utterances_final/frames/processed_frames.pt"
)
save_frames_to_pt(output_folder, parquet_file)
