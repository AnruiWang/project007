import cv2

vc = cv2.VideoCapture('G:/dlc/test/test-eight/video_picture_10/process_label.mp4')
c = 1
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
while rval:
    rval, frame = vc.read()
    cv2.imwrite('G:/dlc/test/test-eight/video_picture_10/original_labeled/'+str(c)+'.bmp', frame)
    c = c+1
    cv2.waitKey(1)
vc.release()

print('转换成功')