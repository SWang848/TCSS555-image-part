from gender_train import Model
from preprocess import Preprocess
import cv2
import os
import pandas as pd

class GenderPrediction:
    def __init__(self, data_path, model_path, model_weight_path):
        self.path = data_path
        self.gender = []
        self.userid = []
        self.model = Model()
        self.model.load_model(model_path, model_weight_path)

    def __switch_gender(self, gender):
        if gender == 0:
            gender = 'male'
        elif gender == 1:
            gender = 'female'
        return gender

    def predict(self):
        for dir_item in os.listdir(self.path):
            if dir_item == 'image':
                full_path = os.path.abspath(os.path.join(self.path, dir_item))
                for image_item in os.listdir(full_path):
                    self.userid.append(image_item[:-4])
                    image = cv2.imread(os.path.join(full_path, image_item))
                    face = Preprocess.crop_faces(image)
                    if len(face) == 0:
                        self.gender.append('female')
                    else:
                        face = Preprocess.resize_image(face, 32, 32)
                        self.gender.append(self.__switch_gender(self.model.gender_predict(face)))

        data = {'userid':self.userid, 'gender':self.gender}
        df = pd.DataFrame(data, index=range(len(self.userid)))

        return df

class AgePrediction:
    def __init__(self, data_path, model_path, model_weight_path):
        self.path = data_path
        self.age = []
        self.userid = []
        self.model = Model()
        self.model.load_model(model_path, model_weight_path)

    def __switch_age(self, age):
        if age == 0:
            age = "xx-24"
        elif age == 1:
            age = "25-34"
        elif age == 2:
            age = "35-49"
        else:
            age = "50-xx"
        return age

    def predict(self):
        for dir_item in os.listdir(self.path):
            if dir_item == 'image':
                full_path = os.path.abspath(os.path.join(self.path, dir_item))
                for image_item in os.listdir(full_path):
                    self.userid.append(image_item[:-4])
                    image = cv2.imread(os.path.join(full_path, image_item))
                    face = Preprocess.crop_faces(image)
                    if len(face) == 0:
                        self.age.append('xx-24')
                    else:
                        face = Preprocess.resize_image(face, 100, 100)
                        self.age.append(self.__switch_age(self.model.gender_predict(face)))

        data = {'userid':self.userid, 'age':self.age}
        df = pd.DataFrame(data, index=range(len(self.userid)))

        return df
