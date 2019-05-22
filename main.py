
IMAGE_SIZE = 256

# 値を-1から1に正規化する関数
def normalize_x(image):
    image = image/127.5 - 1
    return image


# 値を0から1に正規化する関数
def normalize_y(image):
    image = image/255
    return image


# 値を0から255に戻す関数
def denormalize_y(image):
    image = image*255
    return image


# インプット画像を読み込む関数
def load_X(folder_path):
    import os, cv2

    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 3), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        images[i] = normalize_x(image)
    return images, image_files


# ラベル画像を読み込む関数
def load_Y(folder_path):
    import os, cv2

    image_files = os.listdir(folder_path)
    image_files.sort()
    images = np.zeros((len(image_files), IMAGE_SIZE, IMAGE_SIZE, 1), np.float32)
    for i, image_file in enumerate(image_files):
        image = cv2.imread(folder_path + os.sep + image_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image[:, :, np.newaxis]
        images[i] = normalize_y(image)
    return images


#------------------------------------------------------------------------------------


import os
import numpy as np
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from unet import UNet

# ダイス係数を計算する関数
def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


# ロス関数
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# U-Netのトレーニングを実行する関数
def train_unet():
    # trainingDataフォルダ配下にleft_imagesフォルダを置いている
    X_train, file_names = load_X('trainingData' + os.sep + 'left_images')
    # trainingDataフォルダ配下にleft_groundTruthフォルダを置いている
    Y_train = load_Y('trainingData' + os.sep + 'left_groundTruth')

    # 入力はBGR3チャンネル
    input_channel_count = 3
    # 出力はグレースケール1チャンネル
    output_channel_count = 1
    # 一番初めのConvolutionフィルタ枚数は64
    first_layer_filter_count = 64
    # U-Netの生成
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.compile(loss=dice_coef_loss, optimizer=Adam(), metrics=[dice_coef])

    BATCH_SIZE = 12
    # 20エポック回せば十分
    NUM_EPOCH = 20
    history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1)
    model.save_weights('unet_weights.hdf5')


# 学習後のU-Netによる予測を行う関数
def predict():
    import cv2

    # testDataフォルダ配下にleft_imagesフォルダを置いている
    X_test, file_names = load_X('testData' + os.sep + 'left_images')

    input_channel_count = 3
    output_channel_count = 1
    first_layer_filter_count = 64
    network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
    model = network.get_model()
    model.load_weights('unet_weights.hdf5')
    BATCH_SIZE = 12
    Y_pred = model.predict(X_test, BATCH_SIZE)

    for i, y in enumerate(Y_pred):
        # testDataフォルダ配下にleft_imagesフォルダを置いている
        img = cv2.imread('testData' + os.sep + 'left_images' + os.sep + file_names[i])
        y = cv2.resize(y, (img.shape[1], img.shape[0]))
        cv2.imwrite('prediction' + str(i) + '.png', denormalize_y(y))


if __name__ == '__main__':
    train_unet()
    predict()