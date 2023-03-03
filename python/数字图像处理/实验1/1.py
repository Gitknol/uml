from PIL import Image
import numpy as np

import matplotlib.pyplot as plt 
#设置中文
import matplotlib
matplotlib.rc("font",family='YouYuan')
#中文
plt.figure('图片处理')
im_0 = Image.open("python/数字图像处理/实验1/1.png")

#灰度图片
im_1 = Image.open("python/数字图像处理/实验1/1.png").convert("L")
#黑白照片
im_2 = Image.open("python/数字图像处理/实验1/1.png").convert("1")
#im_0.show()
#im_1.show()
#im_2.show()
#im_0.astype(np.int32)
plt.subplot(2,2,1)
plt.imshow(np.array(im_0),cmap='gray')
plt.title('原图')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(np.array(im_1),cmap='gray')
plt.title('灰度图')
plt.xticks([]),plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(np.array(im_2),cmap='gray')
plt.title('二分照片')
plt.xticks([]),plt.yticks([])
#图像保存
plt.imsave("D:/uml/python/数字图像处理/实验1/灰度图.png",im_1,format='png',cmap='gray')
plt.imsave("D:/uml/python/数字图像处理/实验1/二分照片.png",im_2,format='png',cmap='gray')

plt.show()
