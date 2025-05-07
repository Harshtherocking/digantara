import torch 
import torchvision
import PIL 
import os 

from PIL import Image

import torchvision.transforms.functional as F

def preprocess_image(images : list[str] | str )  -> torch.Tensor : 
    """
    min max normalization
    uint16 -> uint8
    brightness = 50 and contrast = 30
    """

    if isinstance(images, str):
        images = [images]

    imgs : list[torch.Tensor] = []

    for image in images : 
        img = Image.open(image)
        img = F.to_tensor(img)
        img = F.center_crop(img, [4418,4418])
        
        # min max normalization
        # min_all not impplemented for UInt16
        min_ = img.min()
        max_ = img.max()
        img = (img - min_) / (max_ - min_) * 255
        img = img.type(torch.uint8)

        #brightness and contrast
        img = F.adjust_contrast(img, 30)
        img = F.adjust_brightness(img, 50)

        imgs.append(img)


    images = torch.stack(imgs)
    assert isinstance(images, torch.Tensor), "cant stack images"
    return images

