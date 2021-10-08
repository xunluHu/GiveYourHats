import cv2

targetPicPath = './facePic2.png'
imageHatPath = './greenHat.png'
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

targetPic = cv2.imread(targetPicPath)
imageHat = cv2.imread(imageHatPath)
rows, cols, channels = imageHat.shape
#颜色空间转成灰度图片
grayTargetPic = cv2.cvtColor(targetPic, cv2.COLOR_BGR2GRAY)
grayHatPic = cv2.cvtColor(imageHat, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(grayHatPic, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

#找出脸所在的位置信息
faces = face_cascade.detectMultiScale(
    grayTargetPic,
    minNeighbors=5,
    minSize=(5, 5)
)
#根据脸所在的位置信息将帽子加入原始图片
if len(faces) > 0:
    for face in faces:
        x, y, w, h = face
        dst = cv2.bitwise_and(targetPic[y - rows:y, x:x + cols], targetPic[y - rows:y, x:x + cols], mask=mask_inv)
        targetPic[y - rows:y, x:x + cols] = cv2.add(dst, imageHat)

cv2.imwrite('./mid.png', grayHatPic)
cv2.imwrite('./test.png', targetPic)
