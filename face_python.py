import cv2

imagePath = r'./facePic2.png'
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

image = cv2.imread(imagePath)
color = (0, 255, 0)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor = 1.15,
    minNeighbors = 5,
    minSize = (5,5)
)

imageHat = cv2.imread('./greenHat.png')
rows, cols, channels = imageHat.shape
#print("rows", rows, "cols", cols, "channels", channels)

#颜色空间转成灰度图片
img2gray = cv2.cvtColor(imageHat, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)
img2_fg = cv2.bitwise_and(imageHat, imageHat, mask = mask)
# cv2.imshow('mask', mask)
# cv2.waitKey(0)

if len(faces) > 0:
    for face in faces:
        x, y, w, h = face
        #cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 3)
        dst = cv2.bitwise_and(image[y - rows:y, x:x + cols], image[y - rows:y, x:x + cols], mask = mask_inv)
        image[y - rows:y, x:x + cols] = cv2.add(dst, img2_fg)

cv2.imwrite(r'./test.png', image)
