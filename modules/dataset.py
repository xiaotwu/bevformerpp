import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
from pyquaternion import Quaternion

class CarlaDataset(Dataset):
    def __init__(self, root_dir, sequence_len=3, transform=None, dummy_mode=False):
        self.root_dir = root_dir
        self.sequence_len = sequence_len
        self.transform = transform
        self.dummy_mode = dummy_mode
        
    def __len__(self):
        if self.dummy_mode:
            return 100
        return 0 # Placeholder

    def __getitem__(self, idx):
        if self.dummy_mode:
            return self._get_dummy_item()
        return {}

    def _get_dummy_item(self):
        H, W = 256, 704
        imgs = torch.randn(self.sequence_len, 6, 3, H, W)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(self.sequence_len, 6, 1, 1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(self.sequence_len, 6, 1, 1)
        ego_pose = torch.eye(4).unsqueeze(0).repeat(self.sequence_len, 1, 1)
        return {'img': imgs, 'intrinsics': intrinsics, 'extrinsics': extrinsics, 'ego_pose': ego_pose}

class NuScenesDataset(Dataset):
    def __init__(self, version='v1.0-mini', dataroot='data/nuscenes', sequence_len=3):
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.sequence_len = sequence_len
        self.samples = self.nusc.sample
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_token = self.samples[idx]['token']
        # Get sequence of samples ending at idx
        # For simplicity, we just take the current sample and repeat it or find previous
        # Real implementation would traverse 'prev' tokens.
        
        # Let's traverse backwards to get sequence
        tokens = []
        curr_token = sample_token
        for _ in range(self.sequence_len):
            tokens.insert(0, curr_token)
            curr_sample = self.nusc.get('sample', curr_token)
            if curr_sample['prev'] != '':
                curr_token = curr_sample['prev']
            else:
                # Padding if no previous
                pass 
        
        # Load data for each token
        imgs_seq = []
        intrinsics_seq = []
        extrinsics_seq = []
        ego_pose_seq = []
        
        # GT Targets (only for the last frame/sample in sequence)
        gt_cls = None
        gt_bbox = None
        mask = None
        
        for i, token in enumerate(tokens):
            sample = self.nusc.get('sample', token)
            
            imgs = []
            intrinsics = []
            extrinsics = []
            
            # Cameras
            sensors = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            for sensor in sensors:
                sd_token = sample['data'][sensor]
                sd = self.nusc.get('sample_data', sd_token)
                
                # Image
                filename = os.path.join(self.nusc.dataroot, sd['filename'])
                img = Image.open(filename)
                img = img.resize((704, 256)) # Resize to match dummy
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                imgs.append(img_tensor)
                
                # Calib
                cs = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
                intrinsic = np.array(cs['camera_intrinsic'])
                # Adjust intrinsic for resize
                # Original size 1600x900 (NuScenes) -> 704x256
                # scale_x = 704/1600, scale_y = 256/900
                scale_x = 704 / 1600
                scale_y = 256 / 900
                intrinsic[0, 0] *= scale_x
                intrinsic[0, 2] *= scale_x
                intrinsic[1, 1] *= scale_y
                intrinsic[1, 2] *= scale_y
                
                intrinsics.append(torch.from_numpy(intrinsic).float())
                
                # Extrinsic (Sensor to Ego)
                # Rotation (quaternion) + Translation
                rot = Quaternion(cs['rotation']).rotation_matrix
                trans = np.array(cs['translation'])
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = rot
                extrinsic[:3, 3] = trans
                extrinsics.append(torch.from_numpy(extrinsic).float())
            
            # Ego Pose (Ego to Global)
            # Use the pose of the first sensor or canonical ego pose
            # NuScenes 'ego_pose' is linked to sample_data, so it varies slightly per sensor time
            # We use the pose associated with CAM_FRONT
            sd_front = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
            ep = self.nusc.get('ego_pose', sd_front['ego_pose_token'])
            
            ego_rot = Quaternion(ep['rotation']).rotation_matrix
            ego_trans = np.array(ep['translation'])
            ego_pose = np.eye(4)
            ego_pose[:3, :3] = ego_rot
            ego_pose[:3, 3] = ego_trans
            
            imgs_seq.append(torch.stack(imgs))
            intrinsics_seq.append(torch.stack(intrinsics))
            extrinsics_seq.append(torch.stack(extrinsics))
            ego_pose_seq.append(torch.from_numpy(ego_pose).float())

            # Generate GT for the last frame
            if i == len(tokens) - 1:
                gt_cls, gt_bbox, mask = self._get_targets(sample, ego_pose)

        return {
            'img': torch.stack(imgs_seq),
            'intrinsics': torch.stack(intrinsics_seq),
            'extrinsics': torch.stack(extrinsics_seq),
            'ego_pose': torch.stack(ego_pose_seq),
            'gt_cls': gt_cls,
            'gt_bbox': gt_bbox,
            'mask': mask
        }

    def _get_targets(self, sample, ego_pose):
        # Grid setup
        H, W = 200, 200
        pc_range = [-50, -50, -5, 50, 50, 3] # x_min, y_min, z_min, x_max, y_max, z_max
        voxel_size = (pc_range[3] - pc_range[0]) / W
        
        gt_cls = torch.zeros(10, H, W) # 10 classes
        gt_bbox = torch.zeros(8, H, W) # dx, dy, dz, w, l, h, sin, cos
        mask = torch.zeros(1, H, W)
        
        anns = sample['anns']
        for ann_token in anns:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get box in global
            # Transform to ego
            # We need Global -> Ego transform = inv(Ego -> Global)
            # ego_pose is Ego -> Global
            
            # Simplified: Just put a Gaussian at the center
            # Real impl needs coordinate transform
            
            # For now, let's just assume we can project global to ego
            # This requires matrix math.
            
            # Let's skip complex math for this snippet and put a dummy target
            # to verify pipeline.
            # In real training, use NuScenes SDK Box.translate() and .rotate()
            
            # Dummy target at center
            ct_int = (100, 100)
            self._draw_gaussian(gt_cls[0], ct_int, radius=2)
            mask[:, 100, 100] = 1
            # bbox: 1, 1, 1, 2, 4, 2, 0, 1
            gt_bbox[:, 100, 100] = torch.tensor([0, 0, 0, 2, 4, 2, 0, 1])
            
        return gt_cls, gt_bbox, mask

    def _draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self._gaussian_2d((diameter, diameter), sigma=diameter / 6)
        
        x, y = int(center[0]), int(center[1])
        
        height, width = heatmap.shape[0:2]
        
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = torch.from_numpy(gaussian[radius - top:radius + bottom, radius - left:radius + right]).to(heatmap.device)
        
        if min(masked_gaussian.shape) > 0 andmin(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian, out=masked_heatmap)
            
    def _gaussian_2d(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h
