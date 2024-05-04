import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

class ShufflePatches(object):
    def __init__(self, patch_size):
        self.ps = patch_size

    def __call__(self, x):
        # divide the batch of images into non-overlapping patches
        u = F.unfold(x, kernel_size=self.ps, stride=self.ps, padding=0)
        # permute the patches of each image in the batch
        permuted_patches = []
        for b_ in u:
            perm = torch.randperm(b_.shape[-1])
            permuted_patch = b_[:, perm]
            permuted_patches.append(permuted_patch)
        pu = torch.cat(permuted_patches, dim=0)
        # fold the permuted patches back together
        f = F.fold(pu, x.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
        return f, perm


import itertools
import random


# Load your input image here
input_image_path = "/workspace/data/ldp-net/source_domain/miniImagenet/n04275548/n04275548_8406.JPEG"
input_image = Image.open(input_image_path)
input_image = TF.crop(input_image, 0, 0, 224, 224)
#input_image.save('/workspace/data/ldp-net/cdfslbenchmark/methods/input_image.jpg')
input_tensor = torch.tensor(np.array(input_image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print(input_tensor.shape)

# Apply the ShufflePatches transformation
shuffled_image_tensor, permut = ShufflePatches(patch_size=112)(input_tensor)
print(permut)
# Convert the tensor back to a PIL image
shuffled_image = Image.fromarray((shuffled_image_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8))

# Save the shuffled image
#output_image_path = "/workspace/data/ldp-net/cdfslbenchmark/methods/shuffled_image.jpg"
#shuffled_image.save(output_image_path)

