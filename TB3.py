# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image as Image

# Load image
image_bgr = cv2.imread('5.jpg')
# Convert to RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Rectange values: start x, start y, width, height
rectangle = (200, 0, 1200, 500)
# Create initial mask
mask = np.zeros(image_rgb.shape[:2], np.uint8)

image_bgr = image_rgb
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
mask = np.zeros(image_rgb.shape[:2], np.uint8)
# Create temporary arrays used by grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Run grabCut
cv2.grabCut(image_rgb, # Our image
            mask, # The Mask
            rectangle, # Our rectangle
            bgdModel, # Temporary array for background
            fgdModel, # Temporary array for background
            5, # Number of iterations
            cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

# Create mask where sure and likely backgrounds set to 0, otherwise 1
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# Multiply image with new mask to subtract background
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
# Show image
plt.imshow(image_rgb_nobg), plt.axis("on")
plt.show()

image_rgb=cv2.imwrite('5_2.png',image_rgb_nobg)

# 以第一個畫素為準，相同色改為透明
def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0,0))
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
    return img

img=Image.open('5_2.png')
img=transparent_back(img)
img.save('5_3.png')


