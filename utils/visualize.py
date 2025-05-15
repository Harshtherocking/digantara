import matplotlib.pyplot as  plt
import numpy as np
import cv2 as cv
import os 
import PIL as pil
from torch import Tensor

DIM = 4418
PATCH_SIZE = 64

def viz_blob (img : np.ndarray, keypoints) :
    patches = []
    centers  = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        half = PATCH_SIZE // 2

        # Boundary check
        if y - half < 0 or y + half > img.shape[0] or x - half < 0 or x + half > img.shape[1]:
            continue

        patch = img[y - half:y + half, x - half:x + half]
        patches.append(patch)
        centers.append((x, y))

    # Show in grid
    n = len(patches) 
    cols = min(6, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.title(f"({centers[i][0]}, {centers[i][1]})", fontsize=8)
        plt.imshow(patches[i], cmap='gray')
        plt.axis('off')
    plt.suptitle("Blobs (Stars + Streaks) - Grid View", fontsize=16)
    plt.tight_layout()
    plt.show()
    pass


def annotate_and_save_image(img : np.ndarray | Tensor, star_keypoints, streak_keypoints, output_path):

    if isinstance(img, Tensor) :
        img = img.numpy()
        img = img.astype(np.uint8)
        img = img * 255

    if len(img.shape) ==  3 and img.shape[0] == 1 :
        img = img[0]
    
    assert(isinstance(img, np.ndarray)), "cant find numpy array"

    # Convert grayscale to BGR
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 1

    for kp in star_keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        cv.circle(img_color, (x, y), r, (0, 255, 255), 1)
        cv.putText(img_color, "Star", (x + 5, y), font, font_scale, (0, 255, 255), thickness)

    for kp in streak_keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        cv.circle(img_color, (x, y), r, (255, 255, 0), 1)
        cv.putText(img_color, "Streak", (x + 5, y), font, font_scale, (255, 255, 0), thickness)

    img_rgb = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)

    # Save annotated image
    output_path = output_path + ".png"
    cv.imwrite(output_path, img_rgb)
    print(f"Annotated image saved to: {output_path}")