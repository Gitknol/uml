import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
def cv_imread(filePath): #读取中文路径的图片
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    #imdecode读取的图像默认会按BGR通道来排列图像矩阵，如果后续需要RGB可以通过如下转换
    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
    return cv_img
#写中文路径图片
def cv_imwrite(filePathName, img):
    try:
        _, cv_img = cv2.imencode(".jpg", img)[1].tofile(filePathName)
        return True
    except:
        return False
#平移
def Move(im,tx,ty,row,cols):
    #生成移动矩阵
    moving_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    im_0 = cv2.warpAffine(im, moving_matrix, (cols, row))
    return im_0
#水平镜像
def Zoom(im,w,row,cols):
    moving_matrix = np.float32([[-1, 0, w], [0, 1, 0]])
    im_0 = cv2.warpAffine(im, moving_matrix, (cols, row))
    return im_0
#垂直镜像
def Zoom_vertical(im,w,row,cols):
    moving_matrix = np.float32([[1, 0, 0], [0, -1, w]])
    im_0 = cv2.warpAffine(im, moving_matrix, (cols, row))
    return im_0
#双线性插值旋转
def Rotate(im,angle1,row,cols):
    pi = math.pi
    angle = angle1 * pi / 180
    matrix_1 = np.array([[1,0,0],[0,-1,0],[-0.5*row,0.5*cols,1]])
    matrix_2 = np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
    matrix_3 = np.array([[1,0,0],[0,-1,0],[0.5*row,0.5*cols,1]])
    im_0 = np.zeros_like(im,dtype=np.uint8)
    for i in range(row):
        for j in range(cols):
            tol_1 = np.matmul(np.array([i,j,1]),matrix_1)
            tol_2 = np.matmul(tol_1,matrix_2)
            tol_3 = np.matmul(tol_2,matrix_3)
            new_i = int(math.floor(tol_3[0]))
            new_j = int(math.floor(tol_3[1]))
            u = tol_3[0] - new_i
            v = tol_3[1] - new_j
            if new_j>=cols or new_i >=row or new_i<1 or new_j<1 or (i+1)>=row or (j+1)>=cols:
                continue

            if (new_i + 1)>=row or (new_j+1)>=cols:
                im_0[i, j, :] = im[new_i,new_j, :]
            else:
                    im_0[i, j] = (1-u)*(1-v)*im[new_i,new_j] + \
                    (1-u)*v*im[new_i,new_j+1] + \
                    u*(1-v)*im[new_i+1,new_j] +\
                    u*v*im[new_i+1,new_j+1]
                





    #生成旋转矩阵
    #moving_matrix = np.float32([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0]])
    #im_0 = cv2.warpAffine(im, moving_matrix, (cols, row))
    return im_0
#最近邻插值放大以及缩小
def Zoom_scale(im,scalew,scaler,row,cols):
    im_0 = np.zeros((scalew,scaler,3),np.uint8)
   # moving_matrix = np.array([[1/scalew, 0, 0], [0, 1/scaler, 0], [0, 0, 1]])
    for i in range(scalew-1):
        for j in range(scaler-1):
            disw = round(i*(row/scalew))
            disc = round(j*(cols/scaler))
            im_0[i][j] = im[disw][disc]         
   # im_0 = cv2.warpAffine(im, moving_matrix, (cols, row))
    return im_0
#缩小   
def Zoom_scale_down(im,scale,row,cols):
    moving_matrix = np.float32([[1/scale, 0, 0], [0, 1/scale, 0]])
    im_0 = cv2.warpAffine(im, moving_matrix, (cols, row))
    return im_0

   
img = cv_imread("python/数字图像处理/实验1/1.png")

#放大
#im_0 = Zoom_scale(img,2000,4000,img.shape[0],img.shape[1])
#平移
#im_0 = Move(img,100,100,img.shape[0],img.shape[1])
#镜像
#im_0 = Zoom_vertical(img,img.shape[0],img.shape[0],img.shape[1])
#im_0 = Zoom(img,img.shape[1],img.shape[0],img.shape[1])
#旋转
im_0 = Rotate(img,180,img.shape[0],img.shape[1])
print(np.array(im_0))
plt.imshow(im_0)
plt.show()