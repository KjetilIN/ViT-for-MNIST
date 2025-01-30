import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Creates the patch embedding before the transformer encoder 

    Args:
        image_size: the size of the image 
        patch_size: size of each patch 
        in_channel: color channels (1 for MNIST)
        embed_dim: dimension of the embedding. 
    """
    def __init__(self, image_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Conv2D does the conversion from a 2d into multiple patches
        # The size of the given patch into the given output dimension, which is the embed_dim
        # See docs: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, h', w')
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x