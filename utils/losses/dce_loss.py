import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

def bilinear_sample(feat_map, coords):
    if coords.shape[0] == 0:
        return torch.empty(0, feat_map.shape[1], device=feat_map.device)
        
    B, C, H, W = feat_map.shape
    N = coords.shape[0]
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    x_norm = 2 * x / (W - 1) - 1
    y_norm = 2 * y / (H - 1) - 1
    
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, N, 1, 2).to(feat_map.device)
    
    sampled = F.grid_sample(feat_map, grid, mode='bilinear', align_corners=True, padding_mode='zeros')
    sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)
    
    return sampled

class DCELoss(nn.Module):
    def __init__(self, mu1=0.40, mu2=0.25, max_intra_pairs=150):
        super().__init__()
        self.mu1 = mu1
        self.mu2 = mu2
        self.max_intra_pairs = max_intra_pairs

    def forward(self, dce_feat, mask, batch_frame_3d_feats, batch_matched_points):
        """
        Args:
            dce_feat: (B, T, C, H, W)
            mask: (B, T, 1, H, W)
            batch_frame_3d_feats: List[List[Array]]
            batch_matched_points: List[List]
        Returns:
            loss_dict: dict
        """
        B, T, C, H, W = dce_feat.shape
        
        total_loss_align = 0.0
        total_loss_dist = 0.0
        total_loss_cross = 0.0
        count_align = 0
        count_dist = 0
        count_cross = 0
        
        def safe_div(a, b):
            return a / b if b > 0 else torch.tensor(0.0, device=dce_feat.device)

        for b_idx in range(B):
            frame_3d_feats = batch_frame_3d_feats[b_idx]
            matched_points = batch_matched_points[b_idx]
            frame_data_cache = []
            
            
            for t_idx in range(T):
                if t_idx >= len(frame_3d_feats) or frame_3d_feats[t_idx].size == 0:
                    frame_data_cache.append(None)
                    continue
                
                frame_data = frame_3d_feats[t_idx]
                coords_t = torch.from_numpy(frame_data[:, :2]).float().to(dce_feat.device)
                emb_3d_t = torch.from_numpy(frame_data[:, 2:]).float().to(dce_feat.device)
                
                z_feat = dce_feat[b_idx:b_idx+1, t_idx]
                m_feat = mask[b_idx:b_idx+1, t_idx]
                
                z_sampled = bilinear_sample(z_feat, coords_t)
                m_sampled = bilinear_sample(m_feat, coords_t)
                valid_mask = (m_sampled > 0.5).squeeze(-1)
                valid_indices = torch.where(valid_mask)[0]
                
                frame_data_cache.append({
                    'z': z_sampled[valid_mask],
                    'e': emb_3d_t[valid_mask]
                })
                
                if len(valid_indices) == 0:
                    continue
                
                loss_align = (z_sampled[valid_mask] - emb_3d_t[valid_mask]).pow(2).sum(dim=-1).mean()
                total_loss_align += loss_align
                count_align += 1
                
                
                num_valid = len(valid_indices)
                if num_valid >= 2:
                    num_pairs = min(self.max_intra_pairs, num_valid * (num_valid - 1) // 2)
                    
                    pairs_np = []
                    attempts = 0
                    max_attempts = num_pairs * 20 
                    
                    while len(pairs_np) < num_pairs and attempts < max_attempts:
                        i, j = random.sample(range(num_valid), 2)
                        if i < j: 
                            pairs_np.append((i, j))
                        attempts += 1
                    
                    if len(pairs_np) > 0:
                        pairs_t = torch.tensor(pairs_np, device=dce_feat.device, dtype=torch.long)
                        i_idx, j_idx = pairs_t[:, 0], pairs_t[:, 1]
                        
                        z_i = z_sampled[valid_mask][i_idx]
                        z_j = z_sampled[valid_mask][j_idx]
                        e_i = emb_3d_t[valid_mask][i_idx]
                        e_j = emb_3d_t[valid_mask][j_idx]
                        
                        d_pixel = torch.norm(z_i - z_j, p=2, dim=-1)
                        d_3d = torch.norm(e_i - e_j, p=2, dim=-1)
                        
                        loss_dist = (d_pixel - d_3d).pow(2).mean()
                        total_loss_dist += loss_dist
                        count_dist += 1
               
            if len(matched_points) > 0:
                loss_cross_sum = 0.0
                valid_count = 0
                
                for point in matched_points:
                    v_idx, f_a, x_a, y_a, f_b, x_b, y_b = point
                    f_a, f_b = int(f_a), int(f_b)
                    
                    if f_a >= len(frame_data_cache) or f_b >= len(frame_data_cache):
                        continue
                    if frame_data_cache[f_a] is None or frame_data_cache[f_b] is None:
                        continue
                    
                    coord_a = torch.tensor([[x_a, y_a]], device=dce_feat.device).float()
                    coord_b = torch.tensor([[x_b, y_b]], device=dce_feat.device).float()
                    
                    z_feat_a = dce_feat[b_idx:b_idx+1, f_a]
                    z_feat_b = dce_feat[b_idx:b_idx+1, f_b]
                    
                    z_a = bilinear_sample(z_feat_a, coord_a)
                    z_b = bilinear_sample(z_feat_b, coord_b)
                    
                    loss_cross_sum += (z_a - z_b).pow(2).sum()
                    valid_count += 1
                
                if valid_count > 0:
                    total_loss_cross += (loss_cross_sum / valid_count)
                    count_cross += 1

        final_align = safe_div(total_loss_align, count_align)
        final_dist = safe_div(total_loss_dist, count_dist)
        final_cross = safe_div(total_loss_cross, count_cross)
        
        total_loss = final_align + self.mu1 * final_dist + self.mu2 * final_cross
        del frame_data_cache
        return {
            'loss_3dgf': total_loss,
            'loss_align': final_align,
            'loss_dist': final_dist,
            'loss_cross': final_cross
        }