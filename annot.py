import os
from utils.preprocess import preprocess_image
from utils.detection import blob_detection
from utils.visualize import annotate_and_save_image


ANNOT_RES_PATH = os.path.join(os.getcwd(), "results", "annotation")

if not os.path.exists(ANNOT_RES_PATH) : 
    os.makedirs(ANNOT_RES_PATH)


def give_name(path)  :
    return"annot_" + "_".join(path.split("\\")[-1].split(".")[0].split("_")[2:])


if __name__ == "__main__": 
    data_path = os.path.join(os.getcwd(), "Datasets", "Raw_Images")

    data = os.listdir(data_path)
    data = [os.path.join(data_path, image) for image in data]

    images = preprocess_image(data)
    print("images preproccessed")

    for org_img_path, img in zip (data, images):
        stars, streaks = blob_detection(img)    
        name = give_name(org_img_path)
        path = os.path.join(ANNOT_RES_PATH, name)
        annotate_and_save_image(img, stars, streaks, path)
    




