import os
import glob
import numpy as np
import cv2

from mmpose.utils import register_all_modules
from mmpose.apis import init_model, inference_topdown

# 1) Register all MMPose modules (required before init_model)
register_all_modules()

# 2) Absolute paths to your config, checkpoint, and data roots:
POSE_CONFIG     = '/home/danielharding/projects/dev/inverse/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
POSE_CHECKPOINT = '/home/danielharding/projects/dev/inverse/mmpose/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
RGB_ROOT        = '/home/danielharding/projects/dev/inverse/dataset/CSL_News/rgb_format'
POSE_OUT_ROOT   = '/home/danielharding/projects/dev/inverse/dataset/CSL_News/pose_format'

# 3) Initialize the pose model on GPU
pose_model = init_model(
    POSE_CONFIG,
    POSE_CHECKPOINT,
    device='cuda:0'
)

# 4) Loop over each video-folder of frames
for folder in sorted(os.listdir(RGB_ROOT)):
    input_dir  = os.path.join(RGB_ROOT, folder)
    output_dir = os.path.join(POSE_OUT_ROOT, folder)
    os.makedirs(output_dir, exist_ok=True)

    # collect all .jpg frames
    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    print(f'Processing "{folder}" with {len(image_paths)} frames')

    # process each frame
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f'[WARN] failed to read {img_path}')
            continue

        # full-frame person box
        person = [{'bbox': [0, 0, img.shape[1], img.shape[0]]}]

        # run top-down pose inference
        pose_results = inference_topdown(
            pose_model,
            img,
            person,
            format='xyxy',
            dataset='TopDownCocoDataset'
        )

        # extract keypoints for the first (and only) person
        if pose_results and 'pred_instances' in pose_results[0]:
            # new API: PoseDataSample -> .pred_instances.keypoints
            keypoints = pose_results[0].pred_instances.keypoints.cpu().numpy()
        else:
            # fallback for older dict-based results
            keypoints = pose_results[0]['keypoints']

        # save as .npy (H, 3) array: (x, y, confidence)
        filename = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        out_path = os.path.join(output_dir, filename)
        np.save(out_path, keypoints)

    print(f'[DONE] saved to {output_dir}')

print('All folders processed.')

