import cv2

avi = cv2.VideoCapture('G:/dlc/test/test-eight/video_picture_8_1/process_label.mp4')
is_opened = avi.isOpened()
print(is_opened)
fps = avi.get(cv2.CAP_PROP_FPS)
print(fps)
widght = avi.get(cv2.CAP_PROP_FRAME_WIDTH)
height = avi.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(str(widght) + 'X' + str(height))
i = 0
while is_opened:
    if i == 300: #截取前300张图片
        break
    else:
        i += 1
    (flag, frame) = avi.read()
    file_name = 'image' + str(i) + '.jpg'
    print(file_name)

    if flag == True:
        cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY])
print('转换完成')