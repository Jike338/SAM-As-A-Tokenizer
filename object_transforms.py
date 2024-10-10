from torchvision import transforms
from torchvision.transforms import functional as F
import torch
import torch.nn as nn
import numpy as np

class RandomResizedCrop (transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=2, antialias=False):
        super().__init__(size, scale, ratio, interpolation, antialias)

    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias), F.resized_crop(mask, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
    
class RandomHorizontalFlip (transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)

    def forward(self, img, mask):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(mask)
        return img, mask
    
class ToTensor (transforms.ToTensor):
    def __init__(self):
        super().__init__()

    def forward(self, img, mask):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img), F.to_tensor(mask)
    
class Normalize (transforms.Normalize):
    def __init__(self, mean, std):
        super().__init__(mean, std)

    def forward(self, img, mask):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(img, self.mean, self.std), mask

class Spalize (nn.Module):
    def __init__(self, size = 224, channels = 3, spalized_channels = 50, object_size_threshold = 50):
        super().__init__()
        
        self.size = size
        self.channels = channels
        self.object_size_threshold = object_size_threshold
        self.spalized_channels = spalized_channels

    def forward(self, img, mask):
        """
        Args:
            img (Tensor): Tensor image to be spalized.
            mask (Tensor): Tensor mask to guide the spalization.
        Returns:
            Tensor: Spalized Tensor image.
        """
        # Initialize tensor to store spalized image
        spalized_img = torch.zeros(self.spalized_channels, self.channels, self.size, self.size)
       
        # Split each object in the mask, and threshold small objects to background value zero
        for obj in range(1, self.spalized_channels):
            obj_mask = (mask == obj)
            # spalize each mask rigion to a spalized channel
            spalized_img[obj] = img * obj_mask
    
        
        #  Assign the background to the zero channel
        spalized_img[0] = img * (mask == 0)
        
        
        return spalized_img, mask



if __name__ == '__main__':
    # Test the spalize transform
    transformer = Spalize()
    input = torch.rand(3, 224, 224)
    mask = torch.randint(0, 256, (1, 224, 224), dtype=torch.uint8)

    print(np.unique(mask.numpy(), return_counts=True))
    
    output = transformer(input, mask)
    
    print(output[0].shape)
    print(np.unique(output[0].numpy(), return_counts=True))
    print(output[1].shape)
    print(np.unique(output[1].numpy(), return_counts=True))
    