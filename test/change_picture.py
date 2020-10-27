import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("G:/dlc/test/test-eight/video_picture_4/change_picture/102.jpg", cv2.IMREAD_COLOR)


cols, rows, _= image.shape
ratio = 0.3
cols = int(ratio*cols)
rows = int(ratio*rows)

image = cv2.resize(image, (rows, cols))

Bch, Gch, Rch = cv2.split(image)


cv2.imshow('Red channel', Rch)
cv2.imwrite('G:/dlc/test/test-eight/video_picture_4/change_picture/Red channel.jpg',Rch)


#红色通道的histgram
#变换程一维向量
pixelSequence=Rch.reshape([rows*cols,])

#统计直方图的组数
numberBins=256

#计算直方图
plt.figure()
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

histogram,bins,patch=plt.hist(pixelSequence,numberBins,facecolor='black',histtype='bar') #facecolor设置为黑色

#设置坐标范围
y_maxValue=np.max(histogram)
plt.axis([0,255,0,y_maxValue])
#设置坐标轴
plt.xlabel("gray Level",fontsize=20)
plt.ylabel('number of pixels',fontsize=20)
plt.title("Histgram of red channel", fontsize=25)
plt.xticks(range(0,255,10))
#显示直方图
plt.pause(0.05)
plt.savefig("G:/dlc/test/test-eight/video_picture_4/change_picture/histgram.png",dpi=260,bbox_inches="tight")
plt.show()


#红色通道阈值
_,RedThresh = cv2.threshold(Rch,110,255,cv2.THRESH_BINARY)

#膨胀操作
element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
erode = cv2.erode(RedThresh, element)

#显示效果
cv2.imshow('original color image',image)
cv2.imwrite('G:/dlc/test/test-eight/video_picture_4/change_picture/scaleimage.jpg',image)

cv2.imshow("RedThresh",RedThresh)
cv2.imwrite('G:/dlc/test/test-eight/video_picture_4/change_picture/RedThresh.jpg',RedThresh)

cv2.imshow("erode",erode)
cv2.imwrite("G:/dlc/test/test-eight/video_picture_4/change_picture/erode.jpg",erode)