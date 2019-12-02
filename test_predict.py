import age_train
from gender_train import Model
import cv2
import os
import pandas as pd
from preprocess import Preprocess
#
# model = Model()
# model.load_model()
#
# image_path = 'test.jpg'
# image = cv2.imread(image_path)
# preprocess = Preprocess()
# face = preprocess.detect_faces(image)
# if len(face) == 0:
#     result = 0
# else:
#     result = model.gender_predict(face)
#
# print(result)

class GenderPrediction:
    def __init__(self, data_path):
        self.path = data_path
        self.preprocess = Preprocess()
        self.gender = []
        self.userid = []
        self.model = Model()
        self.model.load_model()

    def predict(self):
        for dir_item in os.listdir(self.path):
            if dir_item == 'image':
                full_path = os.path.abspath(os.path.join(self.path, dir_item))
                for image_item in os.listdir(full_path):
                    self.userid.append(image_item[:-4])
                    image = cv2.imread(os.path.join(full_path, image_item))
                    face = self.preprocess.detect_faces(image)
                    if len(face) == 0:
                        self.gender.append(1)
                    else:
                        self.gender.append(self.model.gender_predict(face))

        data = {'userid':self.userid, 'gender':self.gender}
        df = pd.DataFrame(data, index=range(len(self.userid)))

        return df

class AgePrediction:
    def __init__(self, data_path, model_path, model_weight_path):
        self.path = data_path
        self.age = []
        self.userid = []
        self.model = age_train.Model()
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
        image = cv2.imread(self.path)
        face = Preprocess.crop_faces(image)
        if len(face) == 0:
            print("xx-24")
        else:
            print(self.model.age_predict(face))
            print(self.__switch_age(self.model.age_predict(face)))



if __name__ == '__main__':
    # test = GenderPrediction('D:\\TCSS555\\project\\public-test-data')
    # result = test.predict()
    # result[result['userid']=='06b055f8e2bca96496514891057913c3']['gender'] = 100
    # print(result[result['userid']=='06b055f8e2bca96496514891057913c3']['gender'])

    test = AgePrediction('./0fa83737d57998508a2bfd8c19e0e17f.jpg', './model/age_model.h5', './model/age_model_80-0.78.hdf5')
    test.predict()