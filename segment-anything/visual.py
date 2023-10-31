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
    for ann in sorted_anns:
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
        # for i in range(3):
        #     img[:, :, i] = color_mask[i]
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

b_json = open('./output/b/grape.json')
b_json_data = json.load(b_json)

large_json = open('./output/large/grape.json')
large_json_data = json.load(large_json)

huge_json = open('./output/huge/grape.json')
huge_json_data = json.load(huge_json)



plt.figure(figsize=(20,10))
# show_anns2(json_data)
plt.subplot(1, 3, 1)
plt.imshow(image)
show_anns2(b_json_data)
plt.title('b_model')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image)
show_anns2(large_json_data)
plt.title('large_model')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image)
show_anns2(huge_json_data)
plt.title('huge_model')
plt.axis('off')
plt.show()
