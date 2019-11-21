import os
import pandas as pd
import cv2
import scipy.io as scio
from preprocess import Preprocess
import numpy

class LoadData:

    def __init__(self):
        self.IMAGE_SIZE = 32
        self.images = []
        self.faces = []
        self.genders = []
        self.ages = []


    def crop_face_fbDataset(self):
        image_path = 'D:\\TCSS555\\project\\training\\image'
        for dir_item in os.listdir(image_path):
            full_path = os.path.abspath(os.path.join(image_path, dir_item))
            image = cv2.imread(full_path)
            # image = upscale(image)
            classifier = cv2.CascadeClassifier(
                "D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=3)
            if len(faceRects) > 0:
                maxFace = 0
                faceMap = {}
                for faceRect in faceRects:
                    x, y, z, h = faceRect
                    faceMap.update({str(z * h): faceRect})
                    maxFace = max(maxFace, z * h)
                mx, my, mz, mh = faceMap[str(maxFace)]
                face = image[my:my + mh + 5, mx:mx + mz + 5]
                face = Preprocess.resize_image(face, self.IMAGE_SIZE, self.IMAGE_SIZE)
                cv2.imwrite('D:\\TCSS555\\project\\training\\faces\\frontal_1.3_3\\{0}'.format(dir_item), face)

            classifier = cv2.CascadeClassifier(
                "D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_profileface.xml")
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=3)
            if len(faceRects) > 0:
                maxFace = 0
                faceMap = {}
                for faceRect in faceRects:
                    x, y, z, h = faceRect
                    faceMap.update({str(z * h): faceRect})
                    maxFace = max(maxFace, z * h)
                mx, my, mz, mh = faceMap[str(maxFace)]
                face = image[my:my + mh + 5, mx:mx + mz + 5]
                face = Preprocess.resize_image(face, self.IMAGE_SIZE, self.IMAGE_SIZE)
                cv2.imwrite('D:\\TCSS555\\project\\training\\faces\\profile_1.3_3\\{0}'.format(dir_item), face)

            image = cv2.flip(image, 1)

            classifier = cv2.CascadeClassifier(
                "D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=3)
            if len(faceRects) > 0:
                maxFace = 0
                faceMap = {}
                for faceRect in faceRects:
                    x, y, z, h = faceRect
                    faceMap.update({str(z * h): faceRect})
                    maxFace = max(maxFace, z * h)
                mx, my, mz, mh = faceMap[str(maxFace)]
                face = image[my:my + mh + 5, mx:mx + mz + 5]
                face = Preprocess.resize_image(face, self.IMAGE_SIZE, self.IMAGE_SIZE)
                cv2.imwrite('D:\\TCSS555\\project\\training\\faces\\flip_frontal_1.3_3\\{0}'.format(dir_item), face)

            classifier = cv2.CascadeClassifier(
                "D:\\opencv\\opencv\\build\\etc\\haarcascades\\haarcascade_profileface.xml")
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceRects = classifier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=3)
            if len(faceRects) > 0:
                maxFace = 0
                faceMap = {}
                for faceRect in faceRects:
                    x, y, z, h = faceRect
                    faceMap.update({str(z * h): faceRect})
                    maxFace = max(maxFace, z * h)
                mx, my, mz, mh = faceMap[str(maxFace)]
                face = image[my:my + mh + 5, mx:mx + mz + 5]
                face = Preprocess.resize_image(face, self.IMAGE_SIZE, self.IMAGE_SIZE)
                cv2.imwrite('D:\\TCSS555\\project\\training\\faces\\flip_profile_1.3_3\\{0}'.format(dir_item), face)

    def load_fbDataset(self, image_path='D:\\TCSS555\\project\\training\\faces'):
        profile = pd.read_csv('D:\\TCSS555\\project\\training\\profile\\profile.csv')
        for dir_item in os.listdir(image_path):
            full_path = os.path.abspath(os.path.join(image_path, dir_item))
            if os.path.isdir(full_path):
                self.load_fbDataset(full_path)
                print(full_path)
            else:
                if dir_item.endswith('.jpg'):
                    image = cv2.imread(full_path)
                    detail = profile[profile['userid'] == dir_item[:-4]]
                    self.faces.append(image)
                    self.genders.append(int(detail['gender']))
                    self.ages.append(Preprocess.switch_age(int(detail['age'])))

        print(len(self.faces), len(self.genders))

        return self.faces, self.genders

    def load_extra_UTKdataset(self, extra_image_path='D:\\TCSS555\\project\\training\\extra_UTK'):
        for dir_item in os.listdir(extra_image_path):
            full_path = os.path.abspath(os.path.join(extra_image_path, dir_item))
            if os.path.isdir(full_path):
                self.load_extra_UTKdataset(full_path)
                print(full_path)
            else:
                if dir_item.endswith('.jpg'):
                    if os.path.basename(full_path).split('_')[1] == '1':
                        image = cv2.imread(full_path)
                        image = Preprocess.resize_image(image, self.IMAGE_SIZE, self.IMAGE_SIZE)
                        self.faces.append(image)
                        self.genders.append(1)
                    elif os.path.basename(full_path).split('_')[1] == '0':
                        image = cv2.imread(full_path)
                        image = Preprocess.resize_image(image, self.IMAGE_SIZE, self.IMAGE_SIZE)
                        self.faces.append(image)
                        self.genders.append(0)

        print(len(self.faces), len(self.genders))

        return self.faces, self.genders


    def load_extra_dataset(self, extra_image_path="D:\\TCSS555\\project\\training\\extra"):
        for dir_item in os.listdir(extra_image_path):
            full_path = os.path.abspath(os.path.join(extra_image_path, dir_item))
            if os.path.isdir(full_path):
                self.load_extra_dataset(full_path)
                print(full_path)
            else:
                if dir_item.endswith('.jpg'):
                    if os.path.basename(os.path.dirname(full_path)) == 'female':
                        image = cv2.imread(full_path)
                        image = Preprocess.resize_image(image, self.IMAGE_SIZE, self.IMAGE_SIZE)
                        self.faces.append(image)
                        self.genders.append(1)
                    elif os.path.basename(os.path.dirname(full_path)) == 'male':
                        image = cv2.imread(full_path)
                        image = Preprocess.resize_image(image, self.IMAGE_SIZE, self.IMAGE_SIZE)
                        self.faces.append(image)
                        self.genders.append(0)

        print(len(self.faces), len(self.genders))

        return self.faces, self.genders

    def make_wikiDataset_label(self):
        data_path = 'D:\\TCSS555\\project\\training\\wiki_crop\\wiki.mat'
        data = scio.loadmat(data_path)

        a = data['wiki'][0][0]
        filenames = []
        genders = []
        for i in range(len(a[2][0])):
            name = a[2][0][i]
            sex = a[3][0][i]
            if numpy.isnan(sex):
                pass
            else:
                sex = 1 if sex==0 else 0
                filenames.append(str(name[0]).split('/')[1])
                genders.append(sex)

        data = {'filename': filenames, 'gender': genders}
        df = pd.DataFrame(data, columns=['filename', 'gender'])
        df.to_csv("D:\\TCSS555\\project\\training\\wiki_crop\\wiki.csv")

    def crop_face_wikiDataset(self):
        extra_image_path = 'D:\\TCSS555\\project\\training\\wiki_crop'
        for dir_item in os.listdir(extra_image_path):
            full_path = os.path.abspath(os.path.join(extra_image_path, dir_item))
            for dir_item in os.listdir(full_path):
                image_full_path = os.path.abspath(os.path.join(full_path, dir_item))
                image = cv2.imread(image_full_path)
                face = Preprocess.crop_faces(image, scaleFactor=1.3, minNeighbors=1)
                if len(face) != 0:
                    cv2.imwrite("D:\\TCSS555\\project\\training\\wiki_faces\\{0}".format(dir_item), face)
                else:
                    pass


    def load_extra_wikiDataset(self):
        image_path = 'D:\\TCSS555\\project\\training\\wiki_faces'
        label = pd.read_csv("D:\\TCSS555\\project\\training\\wiki_crop\\wiki.csv")
        full_path = ''
        for dir_item in os.listdir(image_path):
            full_path = os.path.abspath(os.path.join(image_path, dir_item))
            image = cv2.imread(full_path)
            image = Preprocess.resize_image(image, self.IMAGE_SIZE, self.IMAGE_SIZE)
            gender = label[label['filename'] == dir_item]['gender']
            if gender.empty:
                pass
            else:
                self.faces.append(image)
                self.genders.append(float(gender))

        print(full_path)
        print(len(self.faces), len(self.genders))

        return self.faces, self.genders