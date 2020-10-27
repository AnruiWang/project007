import cv2
import glob

convert_image_path = 'G:/dlc/train/data/batch_1/kuoda/sunbaisen/label/'

fps = 300
size = (512, 512)

videoWriter = cv2.VideoWriter('G:/dlc/test/test-eight/video_picture_10/process_label.mp4',
                              cv2.VideoWriter_fourcc('1', '4', '2', '0'), fps, size)
"""
for i in range(0, 110):
    read_img = cv2.imread(convert_image_path + str(i) + '.bmp')
    videoWriter.write(read_img)
videoWriter.release()
"""


for img in glob.glob(convert_image_path + '/*.bmp'):
    print(img)
    read_img = cv2.imread(img)
    videoWriter.write(read_img)
videoWriter.release()

print('转换成功')