import cv2
import os
import pandas as pd

def upscale(image):
    scale_percent = 100
    width = int(image.shape[1]*scale_percent/100)
    height = int(image.shape[0]*scale_percent/100)
    dim=(width, height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # cv2.imshow('test', resized)
    # cv2.waitKey(10000)
    return resized


def resize_image(image, height, width):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)

    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh -top
    elif w < longest_edge:
        dw = longest_edge - w
        top = dw // 2
        right = dw - left
    else:
        pass

    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    return cv2.resize(constant, (height, width))



profile_path = 'D:\\TCSS555\\project\\training\\profile\\profile.csv'
image_path = 'D:\\TCSS555\\project\\training\\image'
profile = pd.read_csv(profile_path)
# i=0
#
# for dir_item in os.listdir(image_path):
#     i+=1
#     print(i)
#     full_path = os.path.abspath(os.path.join(image_path, dir_item))
#     image = cv2.imread(full_path)
#     # image = upscale(image)
#
#     classifier = cv2.CascadeClassifier("D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
#     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=3)
#     if len(faceRects) > 0:
#         for faceRect in faceRects:
#             maxFace = 0
#             faceMap = {}
#             for faceRect in faceRects:
#                 x, y, z, h = faceRect
#                 faceMap.update({str(z * h): faceRect})
#                 maxFace = max(maxFace, z * h)
#             mx, my, mz, mh = faceMap[str(maxFace)]
#             face = image[my:my + mh+5, mx:mx + mz+5]
#             face = resize_image(face, 200, 200)
#             # cv2.rectangle(image, (x-10, y-10), (x+z+5, y+h+5), (0,255,0), 2)
#         cv2.imwrite('D:\\TCSS555\\project\\training\\test\\frontal_1.3_3\\{0}'.format(dir_item), face)
#
#     classifier = cv2.CascadeClassifier("D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_profileface.xml")
#     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=3)
#     if len(faceRects) > 0:
#         for faceRect in faceRects:
#             maxFace = 0
#             faceMap = {}
#             for faceRect in faceRects:
#                 x, y, z, h = faceRect
#                 faceMap.update({str(z * h): faceRect})
#                 maxFace = max(maxFace, z * h)
#             mx, my, mz, mh = faceMap[str(maxFace)]
#             face = image[my:my + mh+5, mx:mx + mz+5]
#             face = resize_image(face, 200, 200)
#             # cv2.rectangle(image, (x - 10, y - 10), (x + z + 5, y + h + 5), (0, 255, 0), 2)
#         cv2.imwrite('D:\\TCSS555\\project\\training\\test\\profile_1.3_3\\{0}'.format(dir_item), face)
#
#
#     image = cv2.flip(image, 1)
#
#     classifier = cv2.CascadeClassifier("D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
#     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=3)
#     if len(faceRects) > 0:
#         for faceRect in faceRects:
#             maxFace = 0
#             faceMap = {}
#             for faceRect in faceRects:
#                 x, y, z, h = faceRect
#                 faceMap.update({str(z * h): faceRect})
#                 maxFace = max(maxFace, z * h)
#             mx, my, mz, mh = faceMap[str(maxFace)]
#             face = image[my:my + mh+5, mx:mx + mz+5]
#             face = resize_image(face, 200, 200)
#             # cv2.rectangle(image, (x - 10, y - 10), (x + z + 5, y + h + 5), (0, 255, 0), 2)
#         cv2.imwrite('D:\\TCSS555\\project\\training\\test\\flip_frontal_1.3_3\\{0}'.format(dir_item), face)
#
#     classifier = cv2.CascadeClassifier("D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_profileface.xml")
#     grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=3)
#     if len(faceRects) > 0:
#         for faceRect in faceRects:
#             maxFace = 0
#             faceMap = {}
#             for faceRect in faceRects:
#                 x, y, z, h = faceRect
#                 faceMap.update({str(z * h): faceRect})
#                 maxFace = max(maxFace, z * h)
#             mx, my, mz, mh = faceMap[str(maxFace)]
#             face = image[my:my + mh+5, mx:mx + mz+5]
#             face = resize_image(face, 200, 200)
#             # cv2.rectangle(image, (x - 10, y - 10), (x + z + 5, y + h + 5), (0, 255, 0), 2)
#         cv2.imwrite('D:\\TCSS555\\project\\training\\test\\flip_profile_1.3_3\\{0}'.format(dir_item), face)




image = cv2.imread("D:\\TCSS555\\project\\training\\image\\b65f2916ad7399188ebf2c61d64e0828.jpg")
image = cv2.imread("./test.jpg")
# image = upscale(image)
# new_image = cv2.flip(image, 1, dst=None)
def crop_faces(image, scaleFactor=1.3, minNeighbors=1):
    classifier = cv2.CascadeClassifier(
        "D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = classifier.detectMultiScale(grey, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    maxFace = 0
    faceMap = {}
    if len(faceRects) > 0:
        for faceRect in faceRects:
            x, y, z, h = faceRect
            faceMap.update({str(z * h): faceRect})
            maxFace = max(maxFace, z * h)
        mx, my, mz, mh = faceMap[str(maxFace)]
        face = image[my:my + mh, mx:mx + mz]
        # face = Preprocess.resize_image(face, 32, 32)
    else:
        classifier = cv2.CascadeClassifier(
            "D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_profileface.xml")
        faceRects = classifier.detectMultiScale(grey, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, z, h = faceRect
                faceMap.update({str(z * h): faceRect})
                maxFace = max(maxFace, z * h)
            mx, my, mz, mh = faceMap[str(maxFace)]
            face = image[my:my + mh, mx:mx + mz]
            # face = Preprocess.resize_image(face, 32, 32)
        else:
            faceRects = classifier.detectMultiScale(cv2.flip(grey, 1), scaleFactor=scaleFactor,
                                                    minNeighbors=minNeighbors)
            if len(faceRects) > 0:
                for faceRect in faceRects:
                    x, y, z, h = faceRect
                    faceMap.update({str(z * h): faceRect})
                    maxFace = max(maxFace, z * h)
                mx, my, mz, mh = faceMap[str(maxFace)]
                face = image[my:my + mh, mx:mx + mz]
                # face = Preprocess.resize_image(face, 32, 32)
            else:
                return []
    return face

face = crop_faces(image)
cv2.imshow("test", face)
cv2.waitKey(10000)

# classifier = cv2.CascadeClassifier("D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_profileface.xml")
# grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=1)
# print(len(faceRects))
# if len(faceRects) > 0:
#     for faceRect in faceRects:
#         # count += 1
#         x, y, z, h = faceRect
#         face = image[y:y + h, x:x + z]
#         # cv2.rectangle(image, (x, y), (x+z, y+h), (0,255,0), 2)
#         # face = resize_image(face, 200, 200)
#
#     cv2.imshow("test", face)
#     cv2.waitKey(10000)

# if __name__ == '__main__':
#     image = cv2.imread('./test2.jpg')
#     image2 = resize_image(image, 200, 200)
#
#     cv2.imshow('test', image2)
#     cv2.waitKey(100000)