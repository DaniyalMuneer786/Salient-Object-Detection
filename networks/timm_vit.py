""" Vision Transformer (ViT) and Swin Transformer in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

Swin Transformer implementation from:
'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows' - https://arxiv.org/abs/2103.14030

The official jax code is released and available at https://github.com/google-research/vision_transformer
Swin Transformer code from https://github.com/microsoft/Swin-Transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import PatchEmbed, Mlp
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)

class FeatureEnhancement(nn.Module):
    """Feature enhancement module to improve feature quality"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        # Self-attention enhancement
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        # MLP enhancement
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # Swin Transformer models
    'swin_tiny_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    ),
    'swin_small_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    ),
    'swin_base_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
    ),
    'swin_base_patch4_window12_384': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'swin_large_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224.pth',
    ),
    'swin_large_patch4_window12_384': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
}


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 # noel
                 img_size_eval: int = 224):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

        # noel
        self.depth = depth
        self.distilled = distilled
        self.patch_size = patch_size
        self.patch_embed.img_size = (img_size_eval, img_size_eval)

        # Add feature enhancement layers
        self.feature_enhancement = FeatureEnhancement(embed_dim, num_heads)
        
        # Add fusion layers for hierarchical features
        self.fusion_layers = nn.ModuleList([
            nn.Linear(embed_dim * 2, embed_dim) for _ in range(3)
        ])

        # Add pixel-wise feature processing layers
        self.pixel_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.pixel_norm = nn.LayerNorm(embed_dim)
        self.pixel_upsample = nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=True)

        # Add attention refinement module
        self.attention_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 1)
            ) for _ in range(depth)
        ])

        # Add feature fusion module
        self.feature_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim * 2, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(depth-1)
        ])

        # Add spatial attention
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, 1, 7, padding=3),
                nn.Sigmoid()
            ) for _ in range(depth)
        ])

        # Add channel attention
        self.channel_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dim, embed_dim // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 1),
                nn.Sigmoid()
            ) for _ in range(depth)
        ])

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def get_multi_scale_features(self, x):
        """Extract features from multiple layers with enhancement"""
        features = {}
        x = self.prepare_tokens(x)
        
        # Extract features from key layers
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in [3, 7, 11]:  # Early, middle, and late layers
                # Apply feature enhancement
                enhanced_features = self.feature_enhancement(self.norm(x))
                features[f"layer{i+1}"] = enhanced_features
        
        return features

    def get_attention_guided_features(self, x):
        """Extract features guided by attention maps"""
        x = self.prepare_tokens(x)
        attention_maps = []
        features = {}
        
        for i, blk in enumerate(self.blocks):
            # Get attention maps from the block
            if hasattr(blk, 'attn'):
                x, attn = blk.attn(blk.norm1(x), return_attention=True)
                attention_maps.append(attn)
            else:
                x = blk(x)
            
            if i in [3, 7, 11]:  # Key layers
                # Apply attention guidance
                if len(attention_maps) > 0:
                    attn = attention_maps[-1]
                    # Use attention to weight features
                    weighted_features = x * attn.mean(dim=1, keepdim=True)
                    features[f"layer{i+1}"] = self.norm(weighted_features)
                else:
                    features[f"layer{i+1}"] = self.norm(x)
        
        return features

    def get_hierarchical_features(self, x):
        """Extract and fuse hierarchical features"""
        x = self.prepare_tokens(x)
        features = {}
        
        # Bottom-up pathway
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in [3, 7, 11]:  # Key layers
                current_features = self.norm(x)
                if i > 3:
                    # Fuse with previous layer features
                    prev_features = features[f"layer{i-3}"]
                    # Concatenate and fuse
                    fused = torch.cat([prev_features, current_features], dim=-1)
                    features[f"layer{i+1}"] = self.fusion_layers[i//4](fused)
                else:
                    features[f"layer{i+1}"] = current_features
        
        return features

    def get_pixel_wise_features(self, x):
        """Extract pixel-wise features from the transformer"""
        B, C, H, W = x.shape
        x = self.prepare_tokens(x)
        
        # Process through transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in [3, 7, 11]:  # Key layers
                # Get patch tokens (excluding CLS token)
                patch_tokens = x[:, 1:]
                
                # Reshape to spatial dimensions
                h = w = int(math.sqrt(patch_tokens.shape[1]))
                patch_tokens = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)
                
                # Process for pixel-wise features
                pixel_features = self.pixel_conv(patch_tokens)
                pixel_features = self.pixel_norm(pixel_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                
                # Upsample to original image size
                pixel_features = F.interpolate(
                    pixel_features, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=True
                )
                
                # Store features
                self.pixel_features = pixel_features
        
        return self.pixel_features

    def verify_pixel_features(self, x):
        """Verify that pixel-wise features are working correctly"""
        # Get original image dimensions
        B, C, H, W = x.shape
        
        # Get pixel-wise features
        pixel_features = self.get_pixel_wise_features(x)
        
        # Verify dimensions
        assert pixel_features.shape[0] == B, f"Batch size mismatch: {pixel_features.shape[0]} vs {B}"
        assert pixel_features.shape[2] == H, f"Height mismatch: {pixel_features.shape[2]} vs {H}"
        assert pixel_features.shape[3] == W, f"Width mismatch: {pixel_features.shape[3]} vs {W}"
        
        # Print feature statistics
        print(f"Pixel-wise features shape: {pixel_features.shape}")
        print(f"Feature mean: {pixel_features.mean().item():.4f}")
        print(f"Feature std: {pixel_features.std().item():.4f}")
        
        return pixel_features

    def get_dense_pixel_features(self, x):
        """Get dense pixel-wise features with attention guidance"""
        B, C, H, W = x.shape
        x = self.prepare_tokens(x)
        
        # Process through transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in [3, 7, 11]:  # Key layers
                # Get patch tokens and attention
                patch_tokens = x[:, 1:]
                if hasattr(blk, 'attn'):
                    _, attn = blk.attn(blk.norm1(x), return_attention=True)
                    # Reshape attention to spatial dimensions
                    h = w = int(math.sqrt(patch_tokens.shape[1]))
                    attn = attn.mean(dim=1)[:, 1:].reshape(B, h, w)
                    attn = F.interpolate(attn.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=True)
                
                # Reshape tokens to spatial dimensions
                h = w = int(math.sqrt(patch_tokens.shape[1]))
                patch_tokens = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)
                
                # Process for pixel-wise features
                pixel_features = self.pixel_conv(patch_tokens)
                pixel_features = self.pixel_norm(pixel_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                
                # Upsample to original image size
                pixel_features = self.pixel_upsample(pixel_features)
                
                # Apply attention guidance if available
                if 'attn' in locals():
                    pixel_features = pixel_features * attn
                
                # Store features
                self.pixel_features = pixel_features
        
        return self.pixel_features

    def get_enhanced_features(self, x):
        """Extract enhanced features with attention refinement and feature fusion"""
        B, C, H, W = x.shape
        x = self.prepare_tokens(x)
        
        features = {}
        prev_features = None
        
        for i, blk in enumerate(self.blocks):
            # Process through transformer block
            x = blk(x)
            
            # Get patch tokens and reshape
            patch_tokens = x[:, 1:]
            h = w = int(math.sqrt(patch_tokens.shape[1]))
            patch_tokens = patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2)
            
            # Apply attention refinement
            refined = self.attention_refinement[i](patch_tokens)
            
            # Apply spatial attention
            spatial_attn = self.spatial_attention[i](refined)
            refined = refined * spatial_attn
            
            # Apply channel attention
            channel_attn = self.channel_attention[i](refined)
            refined = refined * channel_attn
            
            # Feature fusion with previous layer
            if prev_features is not None:
                refined = self.feature_fusion[i-1](torch.cat([refined, prev_features], dim=1))
            
            # Store features
            features[f"layer{i+1}"] = refined
            prev_features = refined
            
            # Upsample to original size
            if i in [3, 7, 11]:  # Key layers
                features[f"layer{i+1}"] = F.interpolate(
                    features[f"layer{i+1}"],
                    size=(H, W),
                    mode='bilinear',
                    align_corners=True
                )
        
        return features

    def forward(self, x, layer: str = None, encoder_only: bool = False, skip_decoder: bool = False):
        """Enhanced forward pass with improved feature extraction"""
        # Get enhanced features
        enhanced_features = self.get_enhanced_features(x)
        
        # Get multi-scale features
        multi_scale_features = self.get_multi_scale_features(x)
        
        # Get attention-guided features
        attention_features = self.get_attention_guided_features(x)
        
        # Get hierarchical features
        hierarchical_features = self.get_hierarchical_features(x)
        
        # Get pixel-wise features
        pixel_features = self.get_pixel_wise_features(x)
        
        # Get dense pixel features
        dense_pixel_features = self.get_dense_pixel_features(x)
        
        # Verify features if in debug mode
        if hasattr(self, 'debug') and self.debug:
            self.verify_pixel_features(x)
        
        # Combine all features
        features = {
            'enhanced': enhanced_features,
            'multi_scale': multi_scale_features,
            'attention': attention_features,
            'hierarchical': hierarchical_features,
            'pixel_wise': pixel_features,
            'dense_pixel': dense_pixel_features
        }
        
        if encoder_only:
            return features
            
        if layer is not None:
            return {
                'enhanced': enhanced_features.get(layer),
                'multi_scale': multi_scale_features.get(layer),
                'attention': attention_features.get(layer),
                'hierarchical': hierarchical_features.get(layer),
                'pixel_wise': pixel_features.get(layer),
                'dense_pixel': dense_pixel_features.get(layer)
            }
        
        return {
            'enhanced': enhanced_features.get('layer12'),
            'multi_scale': multi_scale_features.get('layer12'),
            'attention': attention_features.get('layer12'),
            'hierarchical': hierarchical_features.get('layer12'),
            'pixel_wise': pixel_features.get('layer12'),
            'dense_pixel': dense_pixel_features.get('layer12')
        }

    # noel - start
    def make_input_divisible(self, x: torch.Tensor):
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.patch_size - W_0 % self.patch_size) % self.patch_size
        pad_h = (self.patch_size - H_0 % self.patch_size) % self.patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=x.mean())

        H_p, W_p = H_0 + pad_h, W_0 + pad_w
        x = nn.functional.pad(x, (0, H_p - W_p, 0, 0) if H_p > W_p else (0, 0, 0, W_p - H_p), value=x.mean())
        return x

    def interpolate_pos_encoding(self, x, pos_embed, size):
        """Interpolate the learnable positional encoding to match the number of patches.

        x: B x (1 + N patches) x dim_embedding
        pos_embed: B x (1 + N patches) x dim_embedding

        return interpolated positional embedding
        """
        npatch = x.shape[1] - 1  # (H // patch_size * W // patch_size)
        N = pos_embed.shape[1] - 1  # 784 (= 28 x 28)
        if npatch == N:
            return pos_embed
        class_emb, pos_embed = pos_embed[:, 0], pos_embed[:, 1:]  # a learnable CLS token, learnable position embeddings

        dim = x.shape[-1]  # dimension of embeddings
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),  # B x dim x 28 x 28
            size=size,
            mode='bicubic',
            align_corners=False
        )

        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        pos_embed = torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)
        return pos_embed

    def prepare_tokens(self, x):
        B, nc, h, w = x.shape
        patch_embed_h, patch_embed_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, self.pos_embed, size=(patch_embed_h, patch_embed_w))
        return self.pos_drop(x)

    def get_tokens(
            self,
            x,
            layers: list,
            patch_tokens: bool = False,
            norm: bool = True,
            input_tokens: bool = False,
            post_pe: bool = False
    ):
        """Return intermediate tokens."""
        list_tokens: list = []

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        if input_tokens:
            list_tokens.append(x)

        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        if post_pe:
            list_tokens.append(x)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)  # B x # patches x dim
            if layers is None or i in layers:
                list_tokens.append(self.norm(x) if norm else x)

        tokens = torch.stack(list_tokens, dim=1)  # B x n_layers x (1 + # patches) x dim

        if not patch_tokens:
            return tokens[:, :, 0, :]  # index [CLS] tokens only, B x n_layers x dim

        else:
            return tokens
    # noel - end


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            trunc_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        trunc_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def swin_tiny_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k"""
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model = _create_swin_transformer('swin_tiny_patch4_window7_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ Compatibility function that returns ViT-Large instead of ViT-Small """
    return vit_large_patch16_224(pretrained=pretrained, **kwargs)


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_tiny_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_deit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_small_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_deit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_base_patch16_224(pretrained=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_deit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_base_patch16_384(pretrained=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_deit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_tiny_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_small_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_base_distilled_patch16_224', pretrained=pretrained,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
    return model