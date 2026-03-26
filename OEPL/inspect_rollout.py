
import argparse
import tarfile
import numpy as np
import imageio
import os
from pathlib import Path
import io
import json

def inspect_rollouts(rollout_dir, output_dir):
    """递归解码 rollout .tar 文件中的所有 video.npy 为 mp4。

    - 支持 rank0/rank1 等子目录（使用 rglob('*.tar') 递归搜索）
    - 每个 .tar 里可能有多条样本：000000000.video.npy / 000000000.meta.json 等
    - 若存在 meta.json，会按 finish_step 裁剪，并在文件名中标记 complete 状态

    Args:
        rollout_dir (str): rollout 根目录（例如 tmp_files/rollout_coffee_eval/train/global_steps=0）。
        output_dir (str): 输出 mp4 根目录。
    """
    rollout_path = Path(rollout_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not rollout_path.is_dir():
        print(f"Error: Directory not found at {rollout_dir}")
        return

    print(f"Searching for .tar files recursively in {rollout_path}...")

    # 递归查找所有 .tar（兼容 rank_0/rank_1 等多层目录）
    tar_files = list(rollout_path.rglob("*.tar"))
    if not tar_files:
        print(f"No .tar files found under {rollout_dir}.")
        return

    print(f"Found {len(tar_files)} .tar files.")

    for tar_path in tar_files:
        rel_tar_dir = tar_path.parent.relative_to(rollout_path)
        # 为每个 tar 单独建一个子目录，方便区分多个样本
        tar_out_dir = output_path / rel_tar_dir / tar_path.stem
        tar_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {tar_path} -> {tar_out_dir}...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                # 找出所有 video.npy
                video_members = [m for m in members if m.name.endswith("video.npy")]
                if not video_members:
                    print(f"  - No 'video.npy' entries in {tar_path.name}. Skipping.")
                    continue

                print(f"  - Found {len(video_members)} videos in {tar_path.name}.")

                for v_member in video_members:
                    # 按 webdataset 命名约定：000000000.video.npy / 000000000.meta.json
                    base = v_member.name[: -len("video.npy")]  # 包含路径前缀
                    sample_id = Path(base).stem  # 例如 '000000000'
                    meta_name = base + "meta.json"
                    meta = None

                    # 加载 video.npy
                    video_file_obj = tar.extractfile(v_member)
                    if video_file_obj is None:
                        print(f"    - Failed to extract {v_member.name}, skipping.")
                        continue

                    video_array = np.load(io.BytesIO(video_file_obj.read()))
                    # 形状预期为 (T, H, W, C)，uint8

                    # 尝试读取 meta.json 以获得 complete / finish_step
                    try:
                        meta_member = tar.getmember(meta_name)
                        meta_file_obj = tar.extractfile(meta_member)
                        if meta_file_obj is not None:
                            meta = json.loads(meta_file_obj.read().decode("utf-8"))
                    except KeyError:
                        meta = None

                    label_suffix = ""
                    if meta is not None:
                        complete = meta.get("complete", None)
                        finish_step = meta.get("finish_step", None)
                        if isinstance(complete, bool):
                            label_suffix = "_succ" if complete else "_fail"
                        # 根据 finish_step 裁剪视频（注意 finish_step 是步数，下标从 0 开始）
                        try:
                            if finish_step is not None:
                                finish_step_int = int(finish_step)
                                if 0 <= finish_step_int < video_array.shape[0]:
                                    # +1 因为 finish_step 是最后一帧的索引
                                    video_array = video_array[: finish_step_int + 1]
                        except Exception:
                            pass

                    print(
                        f"    - Loaded {v_member.name} -> shape {video_array.shape}, "
                        f"complete={meta.get('complete', None) if meta else None}"
                    )

                    # 输出文件名：sample_id[+_succ/_fail].mp4
                    output_video_path = tar_out_dir / f"{sample_id}{label_suffix}.mp4"
                    imageio.mimwrite(output_video_path, video_array, fps=30, quality=8)
                    print(f"    - Saved video: {output_video_path}")

        except Exception as e:
            print(f"  - Failed to process {tar_path}. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rollout .tar files to .mp4 videos.")
    parser.add_argument(
        "rollout_dir",
        type=str,
        help="Path to the directory containing the rollout .tar files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rollout_videos",
        help="Directory to save the generated .mp4 files. Defaults to 'rollout_videos'."
    )
    
    args = parser.parse_args()
    inspect_rollouts(args.rollout_dir, args.output_dir)
