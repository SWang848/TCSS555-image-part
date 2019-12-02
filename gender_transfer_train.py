from tensorflow import keras

from preprocess import Preprocess
from load_dataset import LoadData
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.utils import np_utils
from keras.models import load_model, Model
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Input
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
import math

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

        self.datasets = LoadData()

    def load(self, grey):
        faces, genders = self.datasets.load_fbDataset(grey=grey)
        # faces, genders = self.datasets.load_extra_dataset(grey=grey)
        # faces, genders = self.datasets.load_extra_UTKdataset(grey=grey)
        #faces, genders = self.datasets.load_extra_wikiDataset(grey=grey)
        faces = np.array(faces)
        genders = np.array(genders)

        train_images, valid_images, train_labels, valid_labels = train_test_split(faces, genders, test_size=0.2, random_state=0)
        # train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=0)

        if grey == 1:
            train_images = train_images.reshape(train_images.shape[0], self.datasets.IMAGE_SIZE,
                                                self.datasets.IMAGE_SIZE, 1)
            valid_images = valid_images.reshape(valid_images.shape[0], self.datasets.IMAGE_SIZE,
                                                self.datasets.IMAGE_SIZE, 1)
            # test_images = test_images.reshape(test_images.shape[0], self.datasets.IMAGE_SIZE, self.datasets.IMAGE_SIZE, 3)
            self.input_shape = (self.datasets.IMAGE_SIZE, self.datasets.IMAGE_SIZE, 1)
        else:
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

class _Model:
    def __init__(self, grey):
        self.model = None
        self.hist_fit = None
        self.grey = grey

    def ResNet50_model(self, dataset):


        if self.grey == 1:
            Inp = Input((100, 100, 1))
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 1),
                                  classes=dataset.nb_classes)
        else:
            Inp = Input((100, 100, 3))
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3),
                                  classes=dataset.nb_classes)

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model(Inp)
        x = Flatten()(x)
        predictions = Dense(dataset.nb_classes, activation='sigmoid')(x)
        self.model = Model(inputs=Inp, outputs=predictions)

        self.model.summary()

        return model

    def train(self, data, batch_size=128, nb_epoch=200, data_augmentation=True, file_path='./model/'):
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        # lrate = LearningRateScheduler(self.scheduler)
        lrate = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', factor=0.2, min_lr=0.001)
        self.model.compile(loss='binary_crossentropy', optimizer=adadelta, metrics=['accuracy'])
        checkpoint = ModelCheckpoint(file_path+'model_{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', save_weights_only=True, verbose=1, save_best_only=True, period=5)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=35)

        weights_path = file_path+'model_140-0.76.hdf5'
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print("checkpoint_loaded")

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

            self.hist_fit = self.model.fit_generator(datagen.flow(data.train_images, data.train_labels, batch_size=batch_size),
                                                steps_per_epoch=data.train_images.shape[0]/batch_size,
                                                epochs=nb_epoch,
                                                verbose=1,
                                                validation_data=(data.valid_images, data.valid_labels),
                                                callbacks=[checkpoint, lrate, es])
            # hist_val = self.model.evaluate_generator(datagen.flow(data.valid_images, data.valid_labels, batch_size=batch_size),
            #                                          verbose=1,
            #                                          steps=data.test_images.shape[0]/batch_size)

            with(open('./gender_model_fit_log.txt', 'w+')) as f:
                f.write(str(self.hist_fit.history))

            # with(open('./gender_model_val_log.txt', 'w+')) as f:
            #     f.write(str(hist_val))

    def scheduler(self, epoch):
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(self.model.optimizer.lr)
    # def step_decay(self, epoch):
    #     initial_lrate = 0.1
    #     drop = 0.5
    #     epochs_drop = 10.0
    #     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #     return lrate

    def save_model(self, model_path, model_weight_path):
        self.model.save_weights(model_weight_path)
        self.model.save(model_path)
        print("save finished")

    def load_model(self, model_path, model_weight_path):
        self.model = load_model(model_path)
        self.model.load_weights(model_weight_path)

    def gender_predict(self, image):
        if image.shape != (1, 32, 32, 3):
            image = Preprocess.resize_image(image, 32, 32)
            image = image.reshape((1, 32, 32, 3))

        result = self.model.predict(image)
        print('result:', result[0])

        result = self.model.predict_classes(image)
        gender = result[0]

        return gender

    def visualize_train_history(self):
        print(self.hist_fit.history.keys())
        plt.plot(self.hist_fit.history['acc'])
        plt.plot(self.hist_fit.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_acc', 'val_acc'], loc='upper left')
        plt.savefig('acc_epoch.png')

        plt.plot(self.hist_fit.history['loss'])
        plt.plot(self.hist_fit.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train_loss', 'val_loss'], loc='upper left')
        plt.savefig('loss_epoch.png')

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load(grey=0)

    model = _Model(grey=0)
    model.ResNet50_model(dataset)
    model.train(dataset)
    model.save_model(model_path='./model/gender_model.h5', model_weight_path='./model/gender_model_weight.h5')
    model.visualize_train_history()

    # model = Model()