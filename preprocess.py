import os
import cv2
import pandas as pd

class Preprocess():

    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def switch_age(age):
        if age < 24: return 0
        elif 25 <= age < 34: return 1
        elif 35 <= age < 49: return 2
        else: return 3

    @staticmethod
    def crop_faces(image, scaleFactor=1.3, minNeighbors=3):
        classifier = cv2.CascadeClassifier(
            "D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")
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
                faceRects = classifier.detectMultiScale(cv2.flip(grey, 1), scaleFactor=scaleFactor, minNeighbors=minNeighbors)
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

    @staticmethod
    def upscale(image):
        scale_percent = 100
        width = int(image.shape[1]*scale_percent/100)
        height = int(image.shape[0]*scale_percent/100)
        dim=(width, height)

        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        return resized

    # def load_dataset(self):
    #     image_path = 'D:\\TCSS555\\project\\training\\faces'
    #     profile = pd.read_csv('D:\\TCSS555\\project\\training\\profile\\profile.csv')
    #     for dir_item in os.listdir(image_path):
    #         full_path = os.path.abspath(os.path.join(image_path, dir_item))
    #         if os.path.isdir(full_path):
    #             self.load_dataset(full_path)
    #             print(full_path)
    #         else:
    #             if dir_item.endswith('.jpg'):
    #                 image = cv2.imread(full_path)
    #                 detail = profile[profile['userid'] == dir_item[:-4]]
    #                 self.faces.append(image)
    #                 self.genders.append(int(detail['gender']))
    #                 self.ages.append(self.switch_age(int(detail['age'])))
    #
    #     print(len(self.faces), len(self.genders))
    #
    #     return self.faces, self.genders
    #
    # def load_extra_dataset(self):
    #     extra_image_path = "D:\\TCSS555\\project\\training\\extra"
    #     for dir_item in os.listdir(extra_image_path):
    #         full_path = os.path.abspath(os.path.join(extra_image_path, dir_item))
    #         if os.path.isdir(full_path):
    #             self.load_extra_dataset(full_path)
    #             print(full_path)
    #         else:
    #             if dir_item.endswith('.jpg'):
    #                 if os.path.basename(os.path.dirname(full_path)) == 'female':
    #                     image = cv2.imread(full_path)
    #                     image = self.resize_image(image, 32, 32)
    #                     self.faces.append(image)
    #                     self.genders.append(1)
    #                 elif os.path.basename(os.path.dirname(full_path)) == 'male':
    #                     image = cv2.imread(full_path)
    #                     image = self.resize_image(image, 32, 32)
    #                     self.faces.append(image)
    #                     self.genders.append(0)
    #
    #     print(len(self.faces), len(self.genders))
    #
    #     return self.faces, self.genders
    #
    # def load_extra_UTKdataset(self):
    #     extra_image_path = 'D:\\TCSS555\\project\\training\\extra_UTK'
    #     for dir_item in os.listdir(extra_image_path):
    #         full_path = os.path.abspath(os.path.join(extra_image_path, dir_item))
    #         if os.path.isdir(full_path):
    #             self.load_extra_UTKdataset(full_path)
    #             print(full_path)
    #         else:
    #             if dir_item.endswith('.jpg'):
    #                 if os.path.basename(full_path).split('_')[1] == '1':
    #                     image = cv2.imread(full_path)
    #                     image = self.resize_image(image, 32, 32)
    #                     self.faces.append(image)
    #                     self.genders.append(1)
    #                 elif os.path.basename(full_path).split('_')[1] == '0':
    #                     image = cv2.imread(full_path)
    #                     image = self.resize_image(image, 32, 32)
    #                     self.faces.append(image)
    #                     self.genders.append(0)
    #
    #     print(len(self.faces), len(self.genders))
    #
    #     return self.faces, self.genders

    # def load_extra_wikiDataset(self):
    #     data_path = 'D:\\TCSS555\\project\\training\\wiki_crop\\wiki.mat'
    #     extra_image_path = 'D:\\TCSS555\\project\\training\\wiki_crop'
    #     data = scio.loadmat(data_path)
    #
    #     a = data['wiki'][0][0]
    #     lable = []
    #     for i in range(len(a[2][0])):
    #         name = a[2][0][i]
    #         sex = a[3][0][i]
    #         name = list(name)
    #         name_0 = str(name[0])
    #         sex_float = str(sex)
    #         lable.append((name_0, sex_float))
    #
    #     data = {'filename':lable[0], 'gender':lable[1]}
    #     df = pd.DataFrame(data, columns=['filename','gender'])
    #
    #
    #     for dir_item in os.listdir(extra_image_path):
    #         full_path = os.path.abspath(os.path.join(extra_image_path, dir_item))
    #         for dir_item in full_path:


# if __name__ == '__main__':
#     faces = Preprocess()
#     faces, genders = faces.load_extra_UTKdataset('D:\\TCSS555\\project\\training\\extra_UTK')