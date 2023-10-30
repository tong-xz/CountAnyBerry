import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pycocotools import mask

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    h, w = sorted_anns[0]['segmentation']['size']
    img = np.ones((h, w, 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        rle = {
            'size': ann['segmentation']['size'],
            'counts': ann['segmentation']['counts']
        }
        m = mask.decode(rle)  # This will give you a binary mask
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread('../data/grape_examples/grape.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

file = open('../data/grape_examples/grape.json')
json_data = json.load(file)
print(len(json_data))
print(json_data[0].keys())

# Assuming a default image width and height

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(json_data)

plt.axis('off')
plt.show()