import torch.nn as nn
import torch
from .patch_embeddings import PatchEmbedding

class VisionTransformer(nn.Module):
    """Creates the ViT model with patch embedding, class embedding, positional embedding, Transformer and Multi-head attention."""
    def __init__(
        self,
        image_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=6,
        num_heads=8,
        mlp_ratio=4.,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Class token that represent the classification results
        # This is a parameter which is added to the tunable parameter. 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding as a tunable parameter
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Create the transformer encoder layer 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add class token in front of the patch embeddings 
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Adding th position embedding
        x = x + self.pos_embed
        
        # Transform
        x = self.transformer(x)
        
        # MLP head with the class token 
        # Takes the class token and pass it trough multi-head attention 
        x = x[:, 0] 
        x = self.mlp_head(x)
        
        return x