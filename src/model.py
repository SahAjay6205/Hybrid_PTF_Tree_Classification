import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PointNetBackbone(nn.Module):
    """
    PointNet backbone for global feature encoding.
    This part learns the deep geometric features (F_deep).
    """
    def __init__(self, input_dim=3, global_feat_dim=1024):
        super(PointNetBackbone, self).__init__()
        self.input_dim = input_dim
        self.global_feat_dim = global_feat_dim
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, self.global_feat_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.global_feat_dim)

    def forward(self, x):
        # x shape: (batch_size, num_points, 3)
        # PointNet expects (batch_size, 3, num_points)
        x = x.transpose(2, 1) 
        
        # Pass through shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x_local_features = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x_local_features))
        
        # Max-pooling to get global feature
        # x shape is now (batch_size, 1024, num_points)
        # pool shape will be (batch_size, 1024, 1)
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature = global_feature.view(-1, self.global_feat_dim) # (batch_size, 1024)
        
        return global_feature, x_local_features

class HybridPTF(nn.Module):
    """
    The Hybrid PointNet-Transformer-Fusion (PTF) architecture.
    
    Combines:
    1. PointNet Backbone for global geometric features (F_deep)
    2. Transformer Encoder for non-local context
    3. CNN-based Fusion for structural traits (F_struct)
    """
    def __init__(self, num_classes, num_points=4096, 
                 transformer_dim=128, transformer_heads=4, 
                 transformer_layers=2, struct_trait_dim=3):
        super(HybridPTF, self).__init__()
        
        self.num_points = num_points
        self.struct_trait_dim = struct_trait_dim
        
        # --- 1. Deep Geometric Feature Extraction (F_deep) ---
        
        # PointNet Backbone
        # We use the local features from conv2 (dim 128) for the Transformer
        self.pointnet_backbone = PointNetBackbone(input_dim=3, global_feat_dim=1024)
        
        # Transformer Encoder
        # It operates on the local point features from PointNet (dim 128)
        self.transformer_dim = transformer_dim
        encoder_layer = TransformerEncoderLayer(
            d_model=self.transformer_dim, 
            nhead=transformer_heads,
            dim_feedforward=512,
            batch_first=True # Expects (batch_size, seq_len, features)
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # --- 2. Feature Fusion and Classification ---
        
        # This part combines the two feature streams
        # 1. Global feature from PointNet (1024)
        # 2. Global feature from Transformer (128)
        # 3. Structural traits (3)
        
        fusion_input_dim = 1024 + self.transformer_dim + self.struct_trait_dim
        
        # CNN-based fusion layer (implemented as FC layers)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes) # Final output layer
        )

    def forward(self, x_points, x_struct_traits):
        """
        Forward pass of the Hybrid PTF model.
        
        Args:
            x_points (torch.Tensor): Batch of point clouds (B, K, 3).
            x_struct_traits (torch.Tensor): Batch of structural traits (B, 3).
        """
        
        # --- F_deep (Geometric) ---
        
        # 1. PointNet global feature
        # pointnet_global_feat: (B, 1024)
        # local_point_features: (B, 128, K)
        pointnet_global_feat, local_point_features = self.pointnet_backbone(x_points)
        
        # 2. Transformer non-local feature
        # Transformer expects (B, SeqLen, Features) = (B, K, 128)
        transformer_input = local_point_features.transpose(1, 2)
        
        # transformer_output: (B, K, 128)
        transformer_output = self.transformer_encoder(transformer_input)
        
        # Max-pooling to get a global non-local feature
        # transformer_global_feat: (B, 128)
        transformer_global_feat = torch.max(transformer_output, dim=1)[0]

        # --- F_struct (Ecological) ---
        # x_struct_traits is already (B, 3)
        
        # --- 3. Feature Fusion and Classification ---
        
        # Concatenate (⨁) all features
        f_fusion_input = torch.cat([
            pointnet_global_feat,    # (B, 1024)
            transformer_global_feat, # (B, 128)
            x_struct_traits          # (B, 3)
        ], dim=1)
        
        # Pass through the final classification head
        logits = self.fusion_layer(f_fusion_input)
        
        return logits
