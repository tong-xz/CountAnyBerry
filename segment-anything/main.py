import json
import matplotlib.pyplot as plt
import cv2


image = cv2.imread('images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


file = open('./output/grape.json')
json_data = json.load(file)
print(len(json_data))
print(json_data[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 