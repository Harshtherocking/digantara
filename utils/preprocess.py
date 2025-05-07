import torch 
import cv2 as cv
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

BRIGHTNESS = 50
CONTRAST = 30
DIM = 4418

def preprocess_image(images : list[str] | str )  -> torch.Tensor : 
    """
    cropping
    min max normalization
    uint16 -> uint8
    brightness and contrast
    """

    if isinstance(images, str):
        images = [images]

    imgs : list[torch.Tensor] = []

    for image in images : 
        img = Image.open(image)
        img = np.array(img, dtype=np.uint16)
        
        # min max normalization
        min_ = img.min()
        max_ = img.max()
        img = (img - min_) / (max_ - min_) * 255
        img.astype(np.uint8)

        # brightness and contrast
        img = cv.convertScaleAbs(img, alpha=BRIGHTNESS, beta=CONTRAST)

        img = F.to_tensor(img)
        img = F.center_crop(img, [DIM, DIM])
        imgs.append(img)


    images = torch.stack(imgs)
    assert isinstance(images, torch.Tensor), "cant stack images"
    return images

