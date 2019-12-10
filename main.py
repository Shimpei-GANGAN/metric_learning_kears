#------------------------------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#   Updata History
#   December  08  17:00, 2019 (Sun)
#------------------------------------------------------------

import os, argparse
from pathlib import Path

import numpy as np
from keras.models import Model, load_model, Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

#  パラメータ
#  あとで.yamlファイルに格納する
#train_dir_path		= "./train"
#test_dir_path       = "./test"
#target_size = (224, 224)	#  画像のサイズ

"""
    モデルクラス
"""
class DNN(object):
    #  SGDのパラメータ
    lr = 1e-4					
    momentum = 0.9
    decay = 1e-9
    nesterov = True
    #  ReduceLROnPlateauのパラメータ
    factor              = 1e-1
    min_lr              = 1e-6
    patience            = 3
    
    def __init__(self, args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.input_shape = args.img_size

        #  モデル構築(ここもexistsとかで綺麗に書きたい)
        if Path(".model").glob("*.hdf5") == True:
            model_path = sorted(list(Path("./model").glob("*.hdf5")))
            self.model = load_model( model_path[len(model_path)-1] )
            self.model.load_weights( model_path[len(model_path)-1] )
            print("------------------------------------------")
            print("model_name:{}".format(Path(model_path[len(model_path)-1]).stem))
            print("------------------------------------------")
        else:
            self.model = self.build()
    """
        モデル構築
    """
    def build(self):
        #  VGG16
        vgg16 = VGG16(include_top=False, weights="imagenet", input_shape=self.input_shape)
        #  FCレイヤー
        top_model = Sequential()

        #  モデル構築
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense( 1024, activation="relu" )
        predictions = Dense( 200, activation="relu" )

        self.model = Model(inputs=base_model.input, outputs=predictions)
        #  最適化関数，損失関数
        op = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
        model.compile(loss="mean_squared_error",
                    optimizer=op,
                    metrics=["accuracy"])

    """
        学習フェーズ
    """
    def training(self):
        #  pathlib形式であとで変更
        def ensure_folder(folder):
            if os.path.exists(folder):
                os.makedirs(folder)
        
        train_dir = ensure_folder("data/train/"),
        test_dir = ensure_folder("data/test/"),

        #  学習データ
        train_dategen = ImageDataGenerator(
            rescale=1/255. ,
            horizontal_flip=True,
            vertical_flip=True
        )
        train_gen = train_dategen.flow_from_directory(
            train_dir=train_dir,
            target_size=self.input_shape[:2],
            class_mode="binary",
            batch_size=self.batch_size
        )
        #  テストデータ
        test_dategen = ImageDataGenerator(
            rescale=1/255.
        )
        test_gen = train_dategen.flow_from_directory(
            test_dir=test_dir,
            target_size=self.input_shape[:2],
            class_mode="binary",
            batch_size=self.batch_size
        )

        #  Callbacks Settings
        csv_logger = CSVLogger("./data/training.log")
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=factor,
            patience=patience,
            min_lr=min_lr
        )
        tensor_board = TensorBoard(
            ensure_folder("./logs"),
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )
        check_point = ModelCheckpoint(
            filepath="./model/model.{epoch:02d}-{val_loss:.4f}.hdf5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto"
        )

        cb = [csv_logger, check_point, tensor_board, reduce_lr]

        #  steps_per_epochの設定
        spe = train_dategen.gen_steps(
            train_dir, self.batch_size
        )
        #  validation_stepsの設定
        vs = test_dategen.gen_steps(
            test_dir, self.batch_size
        )

        #  学習
        hist = self.model.fit_generator(
            generator=train_gen,
            steps_per_epoch=spe,
            epochs=self.epochs,
            shuffle=True,
            validation_data=test_gen,
            validation_steps=vs,
            callbacks=cb
        )
        return hist

    """
        モデルの評価
    """
    def predict(self):
        pass

if __name__ == "__main__":
    def parse_args():
        #  parameter
        parser = argparse.ArgumentParser(description="L2 Softmax Model")
        parser.add_argument("--train_dir", default= "./data/train/")
        parser.add_argument("--test_dir", default="./data/test/")
        parser.add_argument("--batch_size", "-b", type=int, default=32,
            				help="Number of images in each mini-batch")
        parser.add_argument("--epochs", "-e", type=int, default=100,
                            help="Number of sweeps over the dataset to train")
        parser.add_argument("--img_size", "-s", type=int, default=224,
                            help="Number of images size")
        args = parser.parse_args()
        return args
    args = parse_args()

    #  Create Model
    md = DNN( args )
    #  Model training
    hist = md.training()
    #plot_history(hist, args.epoch )