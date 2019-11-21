
from preprocess import Preprocess
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorflow.python.keras.models import model_from_json
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D, Conv2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, nb_classes=2):
        self.train_images = None
        self.train_labels = None

        self.valid_images = None
        self.valid_labels = None

        self.test_images = None
        self.test_labels = None

        self.input_shape = None

        self.nb_classes = nb_classes

        self.datasets = Preprocess()

    def load(self):
        self.datasets.load_dataset('D:\\TCSS555\\project\\training\\test')
        extra_faces, extra_genders = self.datasets.load_extra_dataset('D:\\TCSS555\\project\\training\\extra')
        # extra_faces, extra_genders = self.datasets.load_extra_UTKdataset('D:\\TCSS555\\project\\training\\extra_UTK')
        faces = np.array(extra_faces)
        genders = np.array(extra_genders)

        train_images, valid_images, train_labels, valid_labels = train_test_split(faces, genders, test_size=0.1, random_state=0)
        # train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=0)

        train_images = train_images.reshape(train_images.shape[0], self.datasets.IMAGE_SIZE, self.datasets.IMAGE_SIZE, 3)
        valid_images = valid_images.reshape(valid_images.shape[0], self.datasets.IMAGE_SIZE, self.datasets.IMAGE_SIZE, 3)
        # test_images = test_images.reshape(test_images.shape[0], self.datasets.IMAGE_SIZE, self.datasets.IMAGE_SIZE, 3)
        self.input_shape=(self.datasets.IMAGE_SIZE, self.datasets.IMAGE_SIZE, 3)

        print(train_images.shape[0], "train samples")
        print(valid_images.shape[0], 'valid samples')
        # print(test_images.shape[0], 'test samples')

        train_labels = np_utils.to_categorical(train_labels)
        valid_labels = np_utils.to_categorical(valid_labels)
        # test_labels = np_utils.to_categorical(test_labels)
        self.nb_classes = train_labels.shape[1]

        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        # test_images = test_images.astype('float32')

        train_images /= 255
        valid_images /= 255
        # test_images /= 255

        self.train_images = train_images
        self.valid_images = valid_images
        # self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        # self.test_labels = test_labels

class Model:
    def __init__(self):
        self.model = None

    def build_model(self, dataset):
        self.model = Sequential()

        self.model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(dataset.nb_classes))
        self.model.add(Activation('sigmoid'))

        self.model.summary()

    def train(self, data, batch_size=128, nb_epoch=100, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        if not data_augmentation:
            self.model.fit(data.train_images,
                           data.train_labels,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           validation_data=(data.valid_images, data.valid_labels),
                           shuffle=True)
        else:
            datagen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
            )
            datagen.fit(data.train_images)

            hist_fit = self.model.fit_generator(datagen.flow(data.train_images, data.train_labels, batch_size=batch_size),
                                                steps_per_epoch=data.train_images.shape[0]/batch_size,
                                                epochs=nb_epoch,
                                                verbose=1,
                                                validation_data=(data.valid_images, data.valid_labels))
            # hist_val = self.model.evaluate_generator(datagen.flow(data.valid_images, data.valid_labels, batch_size=batch_size),
            #                                          verbose=1,
            #                                          steps=data.test_images.shape[0]/batch_size)

            with(open('./gender_model_fit_log.txt', 'w+')) as f:
                f.write(str(hist_fit.history))

            # with(open('./gender_model_val_log.txt', 'w+')) as f:
            #     f.write(str(hist_val))


    def save_model(self):
        self.model_json = self.model.to_json()
        with open('./model/gender_model_json.json', 'w+') as json_file:
            json_file.write(self.model_json)
        self.model.save_weights('./model/gender_model_weight.h5')
        self.model.save('./model/gender_model.h5')
        print("save finished")

    def load_model(self):
        # json_file = open('./model/gender_model_json.json')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # self.model = model_from_json(loaded_model_json)
        self.model = load_model('./model/gender_model.h5')
        self.model.load_weights('./model/gender_model_weight.h5')

    def gender_predict(self, image):
        preprocess = Preprocess()
        # gender_labels = [0, 1]
        if image.shape != (1, preprocess.IMAGE_SIZE, preprocess.IMAGE_SIZE, 3):
            image = preprocess.resize_image(image, preprocess.IMAGE_SIZE, preprocess.IMAGE_SIZE)
            image = image.reshape((1, preprocess.IMAGE_SIZE, preprocess.IMAGE_SIZE, 3))

        result = self.model.predict(image)
        print('result:', result[0])

        result = self.model.predict_classes(image)
        gender = result[0]

        return gender

    def model_analysis(self, txt_path):
        f = open(txt_path)
        line = f.readline()
        lines = line.split('\'', 7)
        # print(len(lines[].split(",")))

        x_axis = range(100)
        plt.title('Result Analysis')
        # plt.figure(1)
        # plt.subplot(1, 2, 1)
        plt.plot(x_axis, lines[1].split(","), label='validation loss')
        plt.plot(x_axis, lines[5].split(","), label='train_loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()
        #
        # plt.figure(2)
        # plt.subplot(1, 2, 2)
        # plt.plot(x_axis, lines[3].split(","), label='validation acc')
        # plt.plot(x_axis, lines[7].split(","), label='train_acc')
        # plt.legend()
        # plt.xlabel('epoch')
        # plt.ylabel('accuracy')

        plt.savefig("./100_128_graph.png")


        # data = {lines[0]:lines[1].split(","), lines[2]:lines[3].split(","), lines[4]:lines[5].split(","),lines[6]:lines[7].split(",")}
        # df = pd.DataFrame(data, index=[i for i in range(100)])
        # df.to_csv("./gender_model_fit_log.csv ")
        # f.close()

if __name__ == '__main__':
    # dataset = Dataset()
    # dataset.load()

    # model = Model()
    # model.build_model(dataset)
    # model.train(dataset)
    # model.save_model()

    model = Model()
    model.model_analysis('./gender_model_fit_log.txt')