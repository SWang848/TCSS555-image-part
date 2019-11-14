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


if __name__ == '__main__':
    test = GenderPrediction('D:\\TCSS555\\project\\public-test-data')
    result = test.predict()
    result[result['userid']=='06b055f8e2bca96496514891057913c3']['gender'] = 100
    print(result[result['userid']=='06b055f8e2bca96496514891057913c3']['gender'])