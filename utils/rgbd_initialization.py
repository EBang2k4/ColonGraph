"""Initial canonical points from an observed RGB/depth frame, never a GT mesh."""
import hashlib
import math
import numpy as np


def rgbd_to_world(rgb, depth_z, c2w_cv, fov_x, fov_y, max_points=200000):
    rgb = np.asarray(rgb)
    depth = np.asarray(depth_z, dtype=np.float64).squeeze()
    pose = np.asarray(c2w_cv, dtype=np.float64)
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3) or pose.shape != (4, 4):
        raise ValueError('Expected RGB [H,W,3], camera-Z depth [H,W], C2W [4,4]')
    if not np.isfinite(pose).all() or not 0 < fov_x < math.pi or not 0 < fov_y < math.pi:
        raise ValueError('Invalid camera pose/FoV')
    if max_points < 2400: raise ValueError('At least 2400 points required for fixed 2000+400 nodes')
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(rgb).all(-1)
    chosen = np.flatnonzero(valid.ravel())
    if len(chosen) < 2400: raise ValueError('Insufficient valid depth samples')
    if len(chosen) > max_points:
        chosen = chosen[np.linspace(0, len(chosen)-1, max_points, dtype=np.int64)]
    z = depth.ravel()[chosen]
    fx = w / (2 * math.tan(fov_x / 2)); fy = h / (2 * math.tan(fov_y / 2))
    cam = np.stack(((xx.ravel()[chosen]-w/2) * z/fx,
                    (yy.ravel()[chosen]-h/2) * z/fy, z), -1)
    xyz = (np.c_[cam, np.ones(len(cam))] @ pose.T)[:, :3].astype(np.float32)
    colors = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)[chosen].copy()
    if not np.isfinite(xyz).all(): raise ValueError('Non-finite initialized geometry')
    metadata = dict(method='first_train_frame_RGB_GT_depth_backprojection',
                    gt_ply_read=False, initialization_frame_source='train',
                    valid_input_pixels=int(valid.sum()), gaussian_count=len(xyz),
                    depth_convention_used='camera_z_as_in_existing_DynamicColon_loader',
                    physical_mm_conversion_verified=False, coordinate_unit='dataset_scene_unit',
                    fx=fx, fy=fy, cx=w/2, cy=h/2, height=h, width=w,
                    deterministic_point_selection=True,
                    xyz_sha256=hashlib.sha256(xyz.tobytes()).hexdigest(),
                    rgb_sha256=hashlib.sha256(colors.tobytes()).hexdigest())
    return xyz, colors, metadata
