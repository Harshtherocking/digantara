import os
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_image
from utils.detection import blob_detection

data_path = os.path.join(os.getcwd(), "Datasets", "Raw_Images")

data = os.listdir(data_path)
data = [os.path.join(data_path, image) for image in data]

# list = [11, 32, 33, 34]
list  = [11]
data = [data[i] for i in list]

images = preprocess_image(data)
print("images preproccessed")

for img in images :
    stars, streaks = blob_detection(img, plot = True)




