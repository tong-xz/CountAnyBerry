import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pycocotools import mask

def show_anns2(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    a = 0
    for ann in sorted_anns:
        if a == 10:
            break
        rle = {
                        'size': ann['segmentation']['size'],
                        'counts': ann['segmentation']['counts']
                    }

        m = ann['segmentation']
        img = np.ones((m['size'][0], m['size'][1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
            m = mask.decode(rle)
            ax.imshow(np.dstack((img, m*0.35)))
        a = a+1

def show_ann_indi(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    a = 0
    for ann in sorted_anns:
        if a == 10:
            break
        rle = {
            'size': ann['segmentation']['size'],
            'counts': ann['segmentation']['counts']
        }

        m = ann['segmentation']
        img = np.ones((m['size'][0], m['size'][1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        m = mask.decode(rle)

        # 为每一个mask创建新的图像
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(np.dstack((img, m * 0.35)))
        plt.axis('off')
        plt.show()

        a = a + 1




image = cv2.imread('../data/grape_examples/grape.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

file = open('../data/grape_examples/grape.json')
json_data = json.load(file)

show_ann_indi(json_data)

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns2(json_data)
# plt.axis('off')
# plt.show()
