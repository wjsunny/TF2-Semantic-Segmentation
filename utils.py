import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

img = cv2.imread('./images/cat.jpg')
print(img.shape)
image_expanded = np.expand_dims(img, axis=0)
print(image_expanded.shape)
#cv2.imshow('img',img)
#print(img)
#k = cv2.waitKey(0)
#if k == ord('q'):
#    cv2.destroyAllWindows()