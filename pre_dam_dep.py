import os
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from argparse import ArgumentParser

from submodules.depth_anything.depth_anything.util.transform import load_image


session = ort.InferenceSession(
            'submodules/depth_anything/weights/depth_anything_vits14.onnx', 
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )


def get_depth(path):
    image, (orig_h, orig_w) = load_image(path)
    disparity = session.run(None, {"image": image})[0][0, 0]
    disparity = cv2.resize(
        disparity, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC
    )

    # C3VD frames have a black border outside the endoscope's field of view.
    # Treat that region and non-positive disparity as invalid depth rather than
    # converting zero disparity to infinity.
    color = cv2.imread(path, cv2.IMREAD_COLOR)
    valid = (
        (color.max(axis=2) > 3)
        & np.isfinite(disparity)
        & (disparity > 0)
    )
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid] = 1000.0 / disparity[valid]
    return depth


def main(args):
    img_dir = os.path.join(args.dataset_root, args.rgb_paths)
    dam_depth_dir = os.path.join(args.dataset_root, args.output_dir)
    os.makedirs(dam_depth_dir, exist_ok=True)

    img_list = sorted(
        i for i in os.listdir(img_dir)
        if Path(i).suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    print(f"ONNX providers: {session.get_providers()}")
    print(f"Processing {len(img_list)} images from {img_dir} into {dam_depth_dir}")

    started_at = time.perf_counter()
    processed = 0
    skipped = 0
    for frame_idx, i in enumerate(img_list, start=1):
        img_path = os.path.join(img_dir, i)
        dam_depth_path = os.path.join(dam_depth_dir, Path(i).stem + '.npy')
        if os.path.exists(dam_depth_path) and not args.overwrite:
            skipped += 1
            continue

        dam_depth = get_depth(img_path)
        np.save(dam_depth_path, dam_depth)
        processed += 1

        if frame_idx == 1 or frame_idx % 25 == 0 or frame_idx == len(img_list):
            elapsed = time.perf_counter() - started_at
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"[{frame_idx:04d}/{len(img_list):04d}] saved {dam_depth_path} "
                f"shape={dam_depth.shape} rate={rate:.2f} frame/s"
            )

    elapsed = time.perf_counter() - started_at
    print(
        f"Finished: processed={processed}, skipped={skipped}, "
        f"elapsed={elapsed:.2f}s"
    )
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default='data/endonerf/pulling')
    parser.add_argument("--rgb_paths", type=str, default='images')
    parser.add_argument("--output_dir", type=str, default='depth_dam')
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
