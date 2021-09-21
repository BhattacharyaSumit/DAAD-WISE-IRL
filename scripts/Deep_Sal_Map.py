import deepgaze_pytorch
import torch 
import matplotlib.pyplot as plt
import cv2
import pickle
import numpy as np
DEVICE = 'cuda'
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

image = plt.imread('/content/000000487632.jpg')
image = cv2.resize(image, (1024, 1024), interpolation = cv2.INTER_AREA)
centerbias = np.zeros((1024, 1024))

image_tensor = torch.tensor([image.transpose(2, 0, 1)]).to(DEVICE)
centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

log_density_prediction = model(image_tensor, centerbias_tensor)

with open("/content/DG_Sal_Map.txt", "wb") as mfile:
  pickle.dump(log_density_prediction, mfile)

