#!/usr/bin/env python
"""
AlexNet のテストを行います．
"""

import sys

# ROS の OpenCV と干渉する
# pylint: disable=wrong-import-position
WRONG_CV_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if WRONG_CV_PATH in sys.path:
    sys.path.remove(WRONG_CV_PATH)

import os
import re
import numpy as np
import cv2
import argparse
import glob
from matplotlib import pyplot as plt
from lib import dataset
from lib import network
from lib import config

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-r', '--restore', default=None, type=str,
                    metavar='****.h5', help='学習を続きから再開します．チェックポイントの「ファイル名」を入力します．')
PARSER.add_argument('-t', '--train', default=None, type=str, metavar='PATH',
                    help='学習用データセットのディレクトリのパスを入力します．テストのみ行う場合は不要です．(例: "./dataset_train")')
PARSER.add_argument('-v', '--val', default=None, type=str, metavar='PATH',
                    help='確認用データセットのディレクトリのパスを入力します．(例: "./dataset_val")')
PARSER.add_argument('-e', '--test', default="./dataset_test", type=str,
                    metavar='PATH', help='テスト用データセットのディレクトリのパスを入力します．(例: "./dataset_test")')
PARSER.add_argument('-c', '--checkpoint', default=None, type=str,
                    metavar='****.h5', help='テストに用いるチェックポイントを指定する．チェックポイントの「ファイル名」を入力する．')
PARSER.add_argument('-l', '--logdir', default="./logs/test",
                    type=str, metavar='PATH', help='ログ・チェックポイントの出力先を指定する．')
PARSER.add_argument('--batch-size', default=32, type=int,
                    metavar='N', help='学習時のバッチサイズを設定します．')
PARSER.add_argument('--epochs', default=10, type=int,
                    metavar='N', help='学習時のエポック数を設定します．')
PARSER.add_argument('--display', default='off', type=str,
                    choices=["off", "matplotlib", "opencv"], help='描画に使用するライブラリを選択します．')


def main(args):
    """
    みんな大好きメイン関数

    @param args (Namespace): [必須] コマンドラインパーサの入力値が必要です．
    @return None (None): この関数は戻り値がありません．
    """

    # 設定を生成します．
    conf = config.DefaultConfig()
    conf.LOGDIR = args.logdir			# ログ・チェックポイントの出力先
    conf.BATCH_SIZE = args.batch_size  # バッチサイズの指定

    # データセットを生成します．
    if args.test is not None:
        test = dataset.DirectoryBasedDataGenerator(
            config=conf, root=args.test)
        conf.CLASSES = test.classes

    if args.train is not None:
        train = dataset.DirectoryBasedDataGenerator(
            config=conf, root=args.train)               # 訓練用データセットのパスを入力
        conf.CLASSES = train.classes                    # クラスを再設定する必要があります
        if args.val is not None:
            val = dataset.DirectoryBasedDataGenerator(
                config=conf, root=args.val, mode="val")     # 確認用データセットのパスを入力
            assert val.classes == train.classes, "学習データと確認データのクラス数が一致しません．"

    # ネットワークを生成します．
    net = network.AlexNet(conf)

    restore_path = ""
    restore = False
    init_epoch = 0
    if args.restore is not None:
        restore_path = args.restore
        restore = True
    if args.checkpoint is not None:
        restore_path = args.checkpoint
        restore = True
    if restore:
        # 復元します
        net.restore(restore_path)  # チェックポイントの「ファイル名」を入れてください（フルパスではありません）．
        _, checkpoint = os.path.split(restore_path)
        init_epoch = int(re.search("[0-9]{4}", checkpoint).group())

    if args.train is not None:
        # 訓練します．
        if args.val is None:
            val = None
        net.train(train=train, val=val, epochs=args.epochs,
                  initial_epoch=init_epoch, class_weight=train.class_weight)

    test_img_paths = glob.glob(os.path.join(args.test, '*', '*.jpg'))
    test_img_paths.extend(glob.glob(os.path.join(args.test, '*', '*.png')))

    correct = 0.0
    total = 0.0

    for test_img_path in test_img_paths:
        print('')

        # 入力画像を用意します
        image_bgr = cv2.imread(test_img_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 推論します
        pred = net.predict(image_rgb)

        # 表示します
        for i, p in enumerate(pred):
            print("%12s:\t%.2f%%" % (conf.CLASSES[i], p * 100))
        print("%sに写っているのは，%s だと思います．" % (test_img_path, conf.CLASSES[np.argmax(pred)]))

        tag, _ = os.path.split(test_img_path)
        _, tag = os.path.split(tag)

        if tag == conf.CLASSES[np.argmax(pred)]:
            correct += 1.0
        total += 1.0

        if args.display == 'opencv':
            cv2.imshow('Input Image', image_bgr)
            cv2.waitKey(0)
        elif args.display == 'matplotlib':
            plt.imshow(image_rgb)
            plt.show()
        else:
            pass
    
    print('正解率は%.1f％でした．'%(correct / total * 100.0))


if __name__ == '__main__':
    args = PARSER.parse_args()
    assert (args.train is not None or args.checkpoint is not None), 'チェックポイントを指定してください．'
    main(args)
