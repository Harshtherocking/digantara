import os
import torch 
import cv2 as cv
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

BRIGHTNESS = 50
CONTRAST = 30
DIM = 4418

def preprocess_image(images : list[str] | str )  -> torch.Tensor | np.ndarray : 
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

        # print (f"image max : {img.max()}")
        # print (f"image min : {img.min()}")

        img = F.to_tensor(img)
        # changes values to [0, 1]

        # print(f"image maax tensor : {img.max()}")
        # print(f"image min tensor : {img.min()}")

        img = F.center_crop(img, [DIM, DIM])
        imgs.append(img)


    images = torch.stack(imgs)
    assert isinstance(images, torch.Tensor), "cant stack images"
    return images



if  __name__ == "__main__": 
    datapath = os.path.join(os.getcwd(), "Datasets", "Raw_Images")
    data =  os.listdir(datapath)
    data = [os.path.join(datapath, images) for images in data]

    img = preprocess_image(data[4])
    print(img.shape)
