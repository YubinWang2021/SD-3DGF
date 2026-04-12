import pickle
import torch
from torch.utils.data import Dataset

from datasets.utils import *
from config import CONFIG as config
from torchvision import transforms
from datasets.spatial_transforms import GaussianBlurWithProb, ColorJitterWithProb, Scale, ToTensor, Normalize
import random
from collections import defaultdict

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 data_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                ):
        data, self.num_pids = self.read_dataset(data_path)
        self.vertex_embedding_3D = np.load(os.path.join(config.DATA.ROOT, 'vertex_embedding_3D.npy'))
        
        self.spatial_transform = transforms.Compose([
                Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
                ColorJitterWithProb(prob=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlurWithProb(prob=0.5, kernel_size=(11, 11), sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        self.mask_spatial_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=0),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])
        
        self.temporal_transform = temporal_transform
        self.loader = get_loader
        self.dataset = densesampling_for_trainingset(data)
            

    def __len__(self):
        return len(self.dataset)
    
    def read_dataset(self, data_path):
        with open(data_path, 'rb') as f:
            content = pickle.load(f)
        data = content['data']
        num_pids = content['num_pids']
        return data, num_pids
    
    def _resize_coords(self, x, y, orig_w, orig_h, target_w, target_h):

        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        
        new_x = x * scale_w
        new_y = y * scale_h
        
        new_x = np.clip(new_x, 0, target_w - 1)
        new_y = np.clip(new_y, 0, target_h - 1)
        
        return new_x, new_y
    
    def transform_coordinates(self, frame_3d_features, matched_points_list, frame_orig_sizes, target_h, target_w):

        updated_frame_3d_features = []
        num_frames_recorded = len(frame_orig_sizes)
        
        for i, frame_feature in enumerate(frame_3d_features):
            if i >= num_frames_recorded or frame_feature.size == 0:
                updated_frame_3d_features.append(np.array([]) if frame_feature.size == 0 else frame_feature)
                continue
            
            orig_w, orig_h = frame_orig_sizes[i]
            x = frame_feature[:, 0]
            y = frame_feature[:, 1]
            
            scale_w = target_w / orig_w
            scale_h = target_h / orig_h
            new_x = np.clip(x * scale_w, 0, target_w - 1)
            new_y = np.clip(y * scale_h, 0, target_h - 1)
            
            updated_feature = frame_feature.copy()
            updated_feature[:, 0] = new_x
            updated_feature[:, 1] = new_y
            updated_frame_3d_features.append(updated_feature)

        updated_matched_points = []
        for point in matched_points_list:
            v_idx, f_a, x_a, y_a, f_b, x_b, y_b = point


            orig_w_a, orig_h_a = frame_orig_sizes[f_a]
            orig_w_b, orig_h_b = frame_orig_sizes[f_b]
            
            scale_w_a = target_w / orig_w_a
            scale_h_a = target_h / orig_h_a
            new_x_a = float(np.clip(x_a * scale_w_a, 0, target_w - 1))
            new_y_a = float(np.clip(y_a * scale_h_a, 0, target_h - 1))
            
            scale_w_b = target_w / orig_w_b
            scale_h_b = target_h / orig_h_b
            new_x_b = float(np.clip(x_b * scale_w_b, 0, target_w - 1))
            new_y_b = float(np.clip(y_b * scale_h_b, 0, target_h - 1))
            
            updated_point = [v_idx, f_a, new_x_a, new_y_a, f_b, new_x_b, new_y_b]
            updated_matched_points.append(updated_point)


        return updated_frame_3d_features, updated_matched_points
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            img_paths = tracklet['img_paths']
            pid = tracklet['p_id']
            camid = tracklet['cam_id']
            clothes_id = tracklet['clothes_id']
            xcs = tracklet['shape_1024']
            betas = tracklet['betas']

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
            
        """
        tracklet = self.dataset[index]
        (pid, camid, clothes_id, img_paths) = tracklet

        #if self.temporal_transform is not None:
        img_paths_tt = self.temporal_transform(img_paths)
        mask_paths_tt = []
        match_keys = ["CCVID", "VCCR", "SCCVReID", "RCCVReID"]
        frame_3d_features = []
        vertex_observations = defaultdict(dict)  # {vertex_idx: {frame_idx: (x, y)}}

        for frame_idx, img_path in enumerate(img_paths_tt):
            parts = img_path.split('/') 
            joined_path = '_'.join(parts)  
            for key in match_keys:
                if key in img_path:
                    joined_path = joined_path[joined_path.index(key):] 
                    break
            pkl_path = os.path.splitext(joined_path)[0] + ".pkl"
            pkl_path = os.path.join(config.DATA.ROOT, config.DATA.DATASET, 'dense_corr', pkl_path)
            pkl_sampled_path = os.path.splitext(joined_path)[0] + "_samples.pkl"
            pkl_sampled_path = os.path.join(config.DATA.ROOT, config.DATA.DATASET, 'dense_corr', pkl_sampled_path)
            mask_path = os.path.splitext(joined_path)[0] + ".png"
            mask_path = os.path.join(config.DATA.ROOT, config.DATA.DATASET, 'mask', mask_path)
            if config.DATA.DATASET in ['ccvs', 'ccvr']:
                mask_path = img_path.replace("_image_", "_mask_")
            mask_paths_tt.append(mask_path)
            has_pkl = os.path.exists(pkl_path)
            has_sampled = os.path.exists(pkl_sampled_path)
            if not has_pkl and not has_sampled:
                frame_3d_features.append(np.array([]))
                continue
            current_frame_points = []
            if has_pkl:
                with open(pkl_path, 'rb') as f:
                    visible_data = pickle.load(f)  # [M, 3] -> x, y, vertex_idx
                if config.DATA.SAMPLE:
                    max_vis = 500
                    if len(visible_data) > max_vis:
                        indices = np.random.choice(len(visible_data), max_vis, replace=False)
                        visible_data = visible_data[indices]
                visible_xy = visible_data[:, :2]          # (M, 2)
                visible_vidx = visible_data[:, 2].astype(int)  # (M,)



                visible_emb = self.vertex_embedding_3D[visible_vidx]  # (M, D)
                
                visible_feature = np.concatenate([visible_xy, visible_emb], axis=1)
                current_frame_points.append(visible_feature)
                for i in range(len(visible_vidx)):
                    v_idx = visible_vidx[i]
                    vertex_observations[v_idx][frame_idx] = (visible_xy[i, 0], visible_xy[i, 1])
            if has_sampled:
                with open(pkl_sampled_path, 'rb') as f:
                    sampled_data = pickle.load(f)
                sampled_data = np.array(sampled_data)  # (N, 8)
                if config.DATA.SAMPLE:
                    max_sampled = 300
                    if len(sampled_data) > max_sampled:
                        indices = np.random.choice(len(sampled_data), max_sampled, replace=False)
                        sampled_data = sampled_data[indices]
                sample_xy = sampled_data[:, :2]        # (N, 2)
                v012 = sampled_data[:, 2:5].astype(int)  # (N, 3)
                b012 = sampled_data[:, 5:8]              # (N, 3)
                emb0 = self.vertex_embedding_3D[v012[:, 0]]
                emb1 = self.vertex_embedding_3D[v012[:, 1]]
                emb2 = self.vertex_embedding_3D[v012[:, 2]]
                interpolated_emb = emb0 * b012[:, [0]] + emb1 * b012[:, [1]] + emb2 * b012[:, [2]]

                sampled_feature = np.concatenate([sample_xy, interpolated_emb], axis=1)
                current_frame_points.append(sampled_feature)
            if len(current_frame_points) > 0:
                frame_feature = np.vstack(current_frame_points)
            else:
                frame_feature = np.array([])
            frame_3d_features.append(frame_feature)
            
        matched_points_list = []
        vertex_list = list(vertex_observations.keys())
        random.shuffle(vertex_list)

        for v_idx in vertex_list:
            obs_dict = vertex_observations[v_idx]
            frame_list = list(obs_dict.keys())
            
            if len(frame_list) < 2:
                continue
            
            random.shuffle(frame_list)
            f_a, f_b = frame_list[0], frame_list[1]
            x_a, y_a = obs_dict[f_a]
            x_b, y_b = obs_dict[f_b]
            
            matched_points_list.append([v_idx, f_a, x_a, y_a, f_b, x_b, y_b])
            
            max_matches = 50 if config.DATA.SAMPLE else 500
            if len(matched_points_list) >= max_matches:
                break
            
        clip_pil = self.loader(img_paths_tt)
        frame_orig_sizes = []
        for img in clip_pil:
            orig_w, orig_h = img.size 
            frame_orig_sizes.append( (orig_w, orig_h) )

        clip_mask = self.loader(mask_paths_tt)

        clip = [self.spatial_transform(img) for img in clip_pil]
        clip_mask = [self.mask_spatial_transform(mask) for mask in clip_mask]
        
        target_h, target_w = config.DATA.HEIGHT, config.DATA.WIDTH

        updated_frame_3d_features, updated_matched_points = self.transform_coordinates(
            frame_3d_features, 
            matched_points_list, 
            frame_orig_sizes, 
            target_h, 
            target_w
        )
        

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip_mask = torch.stack(clip_mask, 0).permute(1, 0, 2, 3)
        #print(updated_frame_3d_features)
        #print(updated_matched_points)
        return clip, clip_mask, updated_frame_3d_features, updated_matched_points, pid, camid, clothes_id
        
        
        
class TestDataset(Dataset):
    """Video Person ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    """

    def __init__(self,
                 data_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader,
                 seq_len: int=16,
                 stride: int=4
                ):
        data, self.num_pids = self.read_dataset(data_path)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader
        self.dataset, self.vid2clip_index = recombination_for_testset(data, seq_len, stride)
        self.spatial_transform = transforms.Compose([
                Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=3),
                transforms.ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        self.mask_spatial_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            Scale((config.DATA.HEIGHT, config.DATA.WIDTH), interpolation=0),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())
        ])
    def __len__(self):
        return len(self.dataset)
    
    def read_dataset(self, data_path):
        with open(data_path, 'rb') as f:
            content = pickle.load(f)
        data = content['data']
        num_pids = content['num_pids']
        return data, num_pids

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            img_paths = tracklet['img_paths']
            pid = tracklet['p_id']
            camid = tracklet['cam_id']
            clothes_id = tracklet['clothes_id']
            xcs = tracklet['shape_1024']
            betas = tracklet['betas']

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
            
        """
        img_paths, pid, camid, clothes_id = self.dataset[index]     
        
        if self.temporal_transform is not None:
            img_paths_tt = self.temporal_transform(img_paths)
        mask_paths_tt = []
        match_keys = ["CCVID", "VCCR", "SCCVReID", "RCCVReID"]
        for frame_idx, img_path in enumerate(img_paths_tt):
            parts = img_path.split('/') 
            joined_path = '_'.join(parts)  
            for key in match_keys:
                if key in img_path:
                    joined_path = joined_path[joined_path.index(key):] 
                    break
            mask_path = os.path.splitext(joined_path)[0] + ".png"
            mask_path = os.path.join(config.DATA.ROOT, config.DATA.DATASET, 'mask', mask_path)
            if config.DATA.DATASET in ['ccvs', 'ccvr']:
                mask_path = img_path.replace("_image_", "_mask_")
            mask_paths_tt.append(mask_path)

        #clip = self.loader(img_paths_tt)
        clip_pil = self.loader(img_paths_tt)
        frame_orig_sizes = []
        for img in clip_pil:
            orig_w, orig_h = img.size 
            frame_orig_sizes.append( (orig_w, orig_h) )

        clip_mask = self.loader(mask_paths_tt)

        clip = [self.spatial_transform(img) for img in clip_pil]
        clip_mask = [self.mask_spatial_transform(mask) for mask in clip_mask]

        # trans T x C x H x W to C x T x H x W
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip_mask = torch.stack(clip_mask, 0).permute(1, 0, 2, 3)

        return clip, clip_mask, pid, camid, clothes_id
        