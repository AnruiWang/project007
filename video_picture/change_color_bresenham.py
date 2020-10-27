from PIL import Image


i = 1
j = 1
file = "G:/dlc/test/test-eight/video_picture_4"


def change_color_bresenham(z, points):
	img = Image.open(file + "/change_picture/" + str(z+1) + ".jpg")
	for i in range(0, len(points)):
		img.putpixel((points[i][1], points[i][0]), (0, 0, 0))
	img = img.convert("RGB")
	img.save(file + "/change_picture_1/" + str(z+1) + ".jpg")

"""
def change_color(z):
	img = Image.open(file + "/change_picture_1/" + str(z) + ".jpg")

	width = img.size[0]
	height = img.size[1]
	for i in range(0, width):
		for j in range(0, height):
			data = (img.getpixel((i, j)))
			if data[0] < 256 and data[1] < 50 and data[2] < 50:
				img.putpixel((i, j), (0, 0, 0))

	img = img.convert("RGB")
	img.save(file + "/change_picture_1/" + str(z) + ".jpg")


def main():
	for z in range(71, 169):
		print(z)
		change_color(z)


main()
"""