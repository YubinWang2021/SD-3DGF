import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import open_clip
from config import CONFIG

from models.dce_model import DCEModel
from utils.losses.dce_loss import DCELoss


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape

        q = self.q(x_q).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(x_kv).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CLIPVisualWrapper(nn.Module):
    def __init__(self, clip_visual):
        super().__init__()
        self.clip_visual = clip_visual
        self.patch_size = clip_visual.patch_size[0]  # 32

    def forward(self, x):
        B, C, H, W = x.shape
        
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        N_patch_new = H_patch * W_patch
        
        x = self.clip_visual.conv1(x)  # [B, C, 8, 4]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, 32]
        x = x.permute(0, 2, 1)  # [B, 32, C]
        
        cls_token = self.clip_visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # [B, 33, C]

        pos_emb = self.clip_visual.positional_embedding.to(x.dtype).unsqueeze(0)
        cls_pos_emb = pos_emb[:, :1, :]
        patch_pos_emb = pos_emb[:, 1:, :]
        
        N_original_sqrt = int((patch_pos_emb.shape[1]) ** 0.5)
        patch_pos_emb = patch_pos_emb.reshape(1, N_original_sqrt, N_original_sqrt, -1)
        
        patch_pos_emb = F.interpolate(
            patch_pos_emb.permute(0, 3, 1, 2),
            size=(H_patch, W_patch),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)
        
        patch_pos_emb = patch_pos_emb.reshape(1, N_patch_new, -1)
        new_pos_emb = torch.cat([cls_pos_emb, patch_pos_emb], dim=1)
        
        x = x + new_pos_emb
        x = self.clip_visual.patch_dropout(x)
        x = self.clip_visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.clip_visual.transformer(x)
        x = x.permute(1, 0, 2)
        
        patch_features = x[:, 1:, :]
        spatial_feat = patch_features.permute(0, 2, 1).reshape(B, -1, H_patch, W_patch)
        
        return spatial_feat
    
class DINOv2VisualWrapper(nn.Module):
    def __init__(self, dinov2_model):
        super().__init__()
        self.dinov2 = dinov2_model
        self.resize = torch.nn.functional.interpolate

    def forward(self, x):
        if x.shape[-2] % 14 != 0 or x.shape[-1] % 14 != 0:
            new_h = ((x.shape[-2] + 13) // 14) * 14
            new_w = ((x.shape[-1] + 13) // 14) * 14
            x = self.resize(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        outputs = self.dinov2.forward_features(x)
        patch_tokens = outputs['x_norm_patchtokens']  #  [B, N, C]
        
        B, N, C = patch_tokens.shape
        
        H_patch = x.shape[2] // 14
        W_patch = x.shape[3] // 14
        
        assert N == H_patch * W_patch, f"Patch number {N} not match {H_patch}x{W_patch}"
        
        patch_tokens = patch_tokens.permute(0, 2, 1)  # [B, C, N]
        patch_tokens = patch_tokens.reshape(B, C, H_patch, W_patch)  # [B, C, H, W]
        
        return patch_tokens
    
class DREP(nn.Module):
    def __init__(self, dim, num_heads=8, patch_size=(2, 4, 4)): # (t', h', w')
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        
        # MHSA
        self.mhsa_local = MultiHeadAttention(dim, num_heads)
        self.mhsa_global = MultiHeadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, T, H, W = x.shape
        t_p, h_p, w_p = self.patch_size
        

        assert T % t_p == 0 and H % h_p == 0 and W % w_p == 0
        
        x = x.unfold(2, t_p, t_p).unfold(3, h_p, h_p).unfold(4, w_p, w_p)
        x = x.contiguous().view(B, C, -1, t_p * h_p * w_p) # (B, C, Np, S)
        x = x.permute(0, 2, 3, 1).contiguous() # (B, Np, S, C)
        B, Np, S, C = x.shape
        
        R_loc = nn.Parameter(torch.zeros(1, Np, 1, C)).to(x.device)
        R_loc = R_loc.repeat(B, 1, 1, 1) # (B, Np, 1, C)
        
        S_loc_list = []
        for i in range(Np):
            T_pat_i = x[:, i:i+1, :, :] # (B, 1, S, C)
            R_loc_i = R_loc[:, i:i+1, :, :] # (B, 1, 1, C)
            
            # Concat(T_pat_i, R_loc_i): (B, 1, S+1, C)
            T_hat_pat_i = torch.cat([T_pat_i.squeeze(1), R_loc_i.squeeze(1)], dim=1).unsqueeze(1)
            
            # MHSA + Residual
            T_hat_pat_i = T_hat_pat_i.squeeze(1) # (B, S+1, C)
            T_hat_pat_i = self.mhsa_local(self.norm(T_hat_pat_i)) + T_hat_pat_i
            
            updated_R_loc_i = T_hat_pat_i[:, -1:, :] # (B, 1, C)
            S_loc_list.append(updated_R_loc_i)
            
        S_loc = torch.cat(S_loc_list, dim=1) # (B, Np, C)
        

        R_glb = nn.Parameter(torch.zeros(1, 1, C)).to(x.device)
        R_glb = R_glb.repeat(B, 1, 1) # (B, 1, C)
        
        R_hat = torch.cat([S_loc, R_glb], dim=1) # (B, Np+1, C)
        
        # MHSA + Residual
        R_hat = self.mhsa_global(self.norm(R_hat)) + R_hat
        
        R_glb = R_hat[:, -1:, :] # (B, 1, C)
        
        return R_glb, S_loc


class GaitAppearanceFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        
        # Cross-Attention layers
        self.mhca_loc_loc = MultiHeadCrossAttention(dim, num_heads)
        self.mhca_loc_glb = MultiHeadCrossAttention(dim, num_heads)
        self.mhca_glb_glb = MultiHeadCrossAttention(dim, num_heads)
        self.mhca_glb_loc = MultiHeadCrossAttention(dim, num_heads)
        
        self.norm = nn.LayerNorm(dim)
        
        self.w_A = nn.Parameter(torch.zeros(1, 1, dim))
        self.w_G = nn.Parameter(torch.zeros(1, 1, dim))
        self.gamma_weights = nn.Parameter(torch.zeros(1, 4)) # [gamma1, gamma2, gamma3, gamma4]

    def forward(self, R_glb_A, S_loc_A, R_glb_G, S_loc_G):
        B, Np, C = S_loc_A.shape

        S_loc_A_hat = self.mhca_loc_loc(self.norm(S_loc_A), self.norm(S_loc_G)) + \
                      self.mhca_loc_glb(self.norm(S_loc_A), self.norm(R_glb_G)) + S_loc_A
                      
        S_loc_G_hat = self.mhca_loc_loc(self.norm(S_loc_G), self.norm(S_loc_A)) + \
                      self.mhca_loc_glb(self.norm(S_loc_G), self.norm(R_glb_A)) + S_loc_G



        R_glb_A_hat = self.mhca_glb_glb(self.norm(R_glb_A), self.norm(R_glb_G)) + \
                      self.mhca_glb_loc(self.norm(R_glb_A), self.norm(S_loc_G_hat)) + R_glb_A
                      

        R_glb_G_hat = self.mhca_glb_glb(self.norm(R_glb_G), self.norm(R_glb_A)) + \
                      self.mhca_glb_loc(self.norm(R_glb_G), self.norm(S_loc_A_hat)) + R_glb_G


        attn_A = (S_loc_A_hat * self.w_A).sum(dim=-1) # (B, Np)
        alpha = attn_A.softmax(dim=-1).unsqueeze(-1) # (B, Np, 1)
        R_loc_A_bar = (alpha * S_loc_A_hat).sum(dim=1, keepdim=True) # (B, 1, C)
        
        attn_G = (S_loc_G_hat * self.w_G).sum(dim=-1) # (B, Np)
        beta = attn_G.softmax(dim=-1).unsqueeze(-1) # (B, Np, 1)
        R_loc_G_bar = (beta * S_loc_G_hat).sum(dim=1, keepdim=True) # (B, 1, C)
        
        gammas = self.gamma_weights.softmax(dim=-1) # (1, 4)
        fid = gammas[0, 0] * R_loc_A_bar.squeeze(1) + \
              gammas[0, 1] * R_loc_G_bar.squeeze(1) + \
              gammas[0, 2] * R_glb_A_hat.squeeze(1) + \
              gammas[0, 3] * R_glb_G_hat.squeeze(1)
              
        return fid
    
class AppearanceStream(nn.Module):
    def __init__(self, backbone='efficientnet'):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone = CONFIG.MODEL.BACKBONE
        if backbone =='resnet':
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,  # conv2x
                resnet.layer2,  # conv3x
                resnet.layer3   # conv4x 
            )

        elif backbone == 'efficientnet':
            efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            backbone_features = nn.Sequential(*list(efficientnet.features.children())[:5])
            channel_adjust = nn.Sequential(
                nn.Conv2d(80, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )
            self.backbone = nn.Sequential(
                backbone_features,
                channel_adjust
            )

        elif backbone == 'dinov2':
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.backbone = nn.Sequential(
                DINOv2VisualWrapper(dinov2),
                nn.AdaptiveAvgPool2d((16, 8)),
                nn.Conv2d(384, 1024, kernel_size=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )

        elif backbone == 'clip':
            model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            clip_visual = model.visual
            
            if hasattr(clip_visual, 'ln_post'):
                clip_out_dim = clip_visual.ln_post.normalized_shape[0]
            else:
                clip_out_dim = clip_visual.conv1.out_channels
                        
            self.backbone = nn.Sequential(
                CLIPVisualWrapper(clip_visual),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(clip_out_dim, 1024, kernel_size=1, bias=False), 
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()  
        x = x.view(B * T, C, H, W)
        
        feat = self.backbone(x)  
        
        feat = feat.view(B, T, 1024, 16, 8)
        feat = feat.permute(0, 2, 1, 3, 4).contiguous()
        return feat


class GaitStream(nn.Module):

    def __init__(self, dce_chan, ckpt_path, backbone='efficientnet'):
        super().__init__()
        self.dce_model = DCEModel(dce_chan=dce_chan)
        pretrained_path = f"{ckpt_path}/dce_pretrained.pth"
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        self.dce_model.backbone.load_state_dict(checkpoint['backbone'], strict=False)
        self.dce_model.DCEPredictor.load_state_dict(checkpoint['DCEPredictor'], strict=False)
        backbone = CONFIG.MODEL.BACKBONE

        if backbone == 'resnet':
            self.backbone = self._build_resnet()
        elif backbone == 'efficientnet':
            self.backbone = self._build_efficientnet()
        elif backbone == 'dinov2':
            self.backbone = self._build_dinov2()
        elif backbone == 'clip':
            self.backbone = self._build_clip()

        self.lsl_conv1 = nn.Conv3d(26, 256, kernel_size=1, stride=1, padding=0)
        self.lsl_conv2 = nn.Conv3d(1024 + 256, 1024, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
        self.register_buffer('neighbor_offsets', self._get_neighbor_offsets())

    def _build_resnet(self):
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        original_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        with torch.no_grad():
            new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, 64, 1, 1)
            if original_conv1.bias is not None:
                new_conv1.bias.data = original_conv1.bias.data
        resnet.conv1 = new_conv1
        
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
    
    def _build_dinov2(self):
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        original_patch_embed = dinov2.patch_embed.proj
        new_patch_embed = nn.Conv2d(
            in_channels=64,
            out_channels=original_patch_embed.out_channels,
            kernel_size=original_patch_embed.kernel_size,
            stride=original_patch_embed.stride,
            padding=original_patch_embed.padding,
            bias=original_patch_embed.bias is not None
        )
        with torch.no_grad():
            new_patch_embed.weight.data = original_patch_embed.weight.data.mean(dim=1, keepdim=True).repeat(1, 64, 1, 1)
            if original_patch_embed.bias is not None:
                new_patch_embed.bias.data = original_patch_embed.bias.data
        dinov2.patch_embed.proj = new_patch_embed
        
        return nn.Sequential(
            DINOv2VisualWrapper(dinov2),
            nn.AdaptiveAvgPool2d((16, 8)),
            nn.Conv2d(384, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def _build_clip(self):
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_visual = model.visual
        
        if hasattr(clip_visual, 'ln_post'):
            clip_out_dim = clip_visual.ln_post.normalized_shape[0]
        else:
            clip_out_dim = clip_visual.conv1.out_channels

        original_conv1 = clip_visual.conv1
        new_conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        with torch.no_grad():
            new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True).repeat(1, 64, 1, 1)
            if original_conv1.bias is not None:
                new_conv1.bias.data = original_conv1.bias.data
        clip_visual.conv1 = new_conv1
        
        return nn.Sequential(
            CLIPVisualWrapper(clip_visual),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(clip_out_dim, 1024, kernel_size=1, bias=False), 
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
    
    def _build_efficientnet(self):
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        original_stem_container = efficientnet.features[0]
        original_conv = original_stem_container[0]  
        
        new_conv = nn.Conv2d(
            in_channels=64,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        with torch.no_grad():
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, 64, 1, 1)
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data
        
        new_stem_container = nn.Sequential(
            new_conv,
            original_stem_container[1], 
            original_stem_container[2]  
        )
        efficientnet.features[0] = new_stem_container
        
        backbone_features = nn.Sequential(*list(efficientnet.features.children())[:5])
        
        channel_adjust = nn.Sequential(
            nn.Conv2d(80, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        return nn.Sequential(
            backbone_features,
            channel_adjust
        )
    def _get_neighbor_offsets(self):
        offsets = []
        for dt in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dt == 0 and dx == 0 and dy == 0:
                        continue  
                    offsets.append((dt, dx, dy))
        return torch.tensor(offsets, dtype=torch.long)  
    def _compute_lsl(self, feat):

        B, C, T, H, W = feat.shape
        K = len(self.neighbor_offsets)  


        feat_padded = F.pad(feat, pad=(1, 1, 1, 1, 1, 1), mode='constant', value=0.0)

        feat_norm = F.normalize(feat, dim=1, eps=1e-12)
        feat_padded_norm = F.normalize(feat_padded, dim=1, eps=1e-12)
        

        similarity = torch.zeros(B, K, T, H, W, device=feat.device, dtype=feat.dtype)


        for i, (dt, dx, dy) in enumerate(self.neighbor_offsets):
            t_start = 1 + dt
            t_end = 1 + dt + T
            h_start = 1 + dx
            h_end = 1 + dx + H
            w_start = 1 + dy
            w_end = 1 + dy + W

            neighbor_feat_view = feat_padded_norm[:, :, t_start:t_end, h_start:h_end, w_start:w_end]

            # (B, C, T, H, W) * (B, C, T, H, W) -> sum over C -> (B, T, H, W)
            sim_i = (feat_norm * neighbor_feat_view).sum(dim=1)
            
            similarity[:, i:i+1, :, :, :] = sim_i.unsqueeze(1)

        return similarity

    def forward(self, x, mask):
        B, C, T, H, W = x.shape
        
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 3, H, W)
        x = x.view(B * T, C, H, W)
        
        mask = mask.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, 1, H, W)
        mask = mask.view(B * T, 1, H, W)
        
        dce_feat = self.dce_model(x)  # (B*T, 64, 256, 128)
        dce_feat_5d = dce_feat.view(B, T, 64, 256, 128)
        dce_feat_for_backbone = dce_feat.detach()
        mask_5d = mask.view(B, T, 1, 256, 128) 
        mask_for_backbone = mask.repeat(1, dce_feat.shape[1], 1, 1)  
        dce_feat_for_backbone = dce_feat_for_backbone * mask_for_backbone  
        
        feat = self.backbone(dce_feat_for_backbone)  # (B*T, 1024, 16, 8)
        feat = feat.view(B, T, 1024, 16, 8)
        feat = feat.permute(0, 2, 1, 3, 4).contiguous()
        
        lsl_feat = self._compute_lsl(feat)  # (B, 26, T, 16, 8)
        lsl_feat_up = self.relu(self.lsl_conv1(lsl_feat))  # (B, 256, T, 16, 8)
        concat_feat = torch.cat([feat, lsl_feat_up], dim=1)  # (B, 1024+256, T, 16, 8)
        gait_feat = self.lsl_conv2(concat_feat)  # (B, 1024, T, 16, 8)
        
        return gait_feat, dce_feat_5d, mask_5d


class SD_3DGF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.appearance_stream = AppearanceStream()
        self.gait_stream = GaitStream(config.MODEL.DCE_CHAN, ckpt_path=config.METADATA.PRETRAINED_PATH)
        self.feat_dim = config.MODEL.APP_FEATURE_DIM
        self.drep_A = DREP(dim=self.feat_dim, patch_size=(2, 4, 4)) 
        self.drep_G = DREP(dim=self.feat_dim, patch_size=(2, 4, 4))
        self.fusion = GaitAppearanceFusion(dim=self.feat_dim)
        self.dce_loss = DCELoss(mu1=config.LOSS.CST1_LOSS_WEIGHT, mu2=config.LOSS.CST1_LOSS_WEIGHT, max_intra_pairs=150)
        classifier_path = f"{config.METADATA.PRETRAINED_PATH}/{config.DATA.DATASET}_clothes_classifier.pth"
        state_dict = torch.load(classifier_path, map_location='cpu')
        out_dim, in_dim = state_dict['classifier.weight'].shape
        classifier_model = nn.Linear(in_dim, out_dim)
        state_dict = {k.replace('classifier.', ''): v for k, v in state_dict.items()}
        classifier_model.load_state_dict(state_dict, strict=True)
        classifier_model.eval()
        for param in classifier_model.parameters():
            param.requires_grad = False
        self.register_buffer('Wc', classifier_model.weight.data.detach())
        



    def forward(self, clip, clip_mask, frame_3d_feats=None, matched_points=None, mode='train'):

        
        app_feat = self.appearance_stream(clip)
        gait_feat, dce_feat_5d, mask_5d = self.gait_stream(clip, clip_mask)
        R_glb_A, S_loc_A = self.drep_A(app_feat)
        R_glb_G, S_loc_G = self.drep_G(gait_feat)
        
        fid = self.fusion(R_glb_A, S_loc_A, R_glb_G, S_loc_G)
        
        if mode == 'train': 
            loss_dict = {}
            if frame_3d_feats is not None:
                loss_dict.update(self.dce_loss(dce_feat_5d, mask_5d, frame_3d_feats, matched_points))
                fA = F.adaptive_avg_pool3d(app_feat, (1, 1, 1)).flatten(1)
                projection = fA @ self.Wc.T.detach()
                loss_dict['loss_org'] = -torch.mean(torch.sum(projection ** 2, dim=1))
            return fid, loss_dict
        else:
            return fid

def get_SD_3DGF(config, **kwargs):
    return SD_3DGF(config=config,
                      **kwargs)

if __name__ == "__main__":
    efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    features = list(efficientnet.features.children())
    
    dummy_input = torch.randn(1, 3, 256, 128)
    x = dummy_input

