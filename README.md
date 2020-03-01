# AlexNet

このリポジトリは学生の練習用です．
動作の保証は致しません．

## セットアップ
はじめにこのリポジトリをクローンしてください．
```sh
$ git clone https://github.com/Meijo-TasakiLab/AlexNet.git
```
**Google Colab を使う場合は以上で終了です．**

次のコマンドを実行してください．
```sh
sudo apt install -y python3-tk
mkdir ~/virtualenv
virtualenv --no-site-package -p python3 ~/virtualenv/alexnet
source ~/virtualenv/alexnet/bin/activate
pip install imgaug==0.2.9 opencv-python tensorflow-gpu==1.12.0 numpy glob3 matplotlib
```

## 実行方法

1. 下のように学習データ・確認データ・テストデータを配置します．  
(例)  
├── dataset_train（学習データ）  
│   ├── dog  
│   │   ├── dog_train_0001.png  
│   │   ├── dog_train_0002.png  
│   │   ├── ：  
│   ├── cat  
│   │   ├── cat_train_0001.png  
│   │   ├── cat_train_0002.png  
│   │   ├── ：  
├── dataset_val（確認データ）  
│   ├── dog  
│   │   ├── dog_val_0001.png  
│   │   ├── dog_val_0002.png  
│   │   ├── ：  
│   ├── cat  
│   │   ├── cat_val_0001.png  
│   │   ├── cat_val_0002.png  
│   │   ├── ：  
├── dataset_test（テストデータ）  
│   ├── dog  
│   │   ├── dog_test_0001.png  
│   │   ├── dog_test_0002.png  
│   │   ├── ：  
│   ├── cat  
│   │   ├── cat_test_0001.png  
│   │   ├── cat_test_0002.png  
│   │   ├── ：  

2. 次のコマンドを実行して仮想環境に入ります(Google Colab では不要です)．
    ```sh
    source ~/virtualenv/alexnet/bin/activate
    ```

3. 次のコマンドを実行することで学習・テストを行います．  
    (例)
    ```sh
    ./alexnet.py -t dataset_train/ -v dataset_val/ -e dataset_test/
    ```
    その他のオプション
    * --epochs [整数] : 学習する際のエポック数を指定します．
    * --batch-size [整数] : 学習する際のバッチサイズを指定します．
    * --display ["off" or "matplotlib" or "opencv"] : 画像を表示する際のライブラリを指定します．(既定: "off")

### 途中から実行するには
次のオプションを指定すると途中から実行されます．
* -l [PATH] : チェックポイントあるディレクトリのパスを指定します．
* --restore [****.h5] : チェックポイントの「ファイル名」を入力します．

### テストのみ実行するには
次のオプションを指定するとテストのみ実行されます．
* -e [PATH] : テストデータを配置したディレクトリを指定します．
* -l [PATH] : チェックポイントあるディレクトリのパスを指定します．
* --checkpoint [****.h5] : チェックポイントの「ファイル名」を入力します．
* --display ["matplotlib" or "opencv"] : 画像を表示する際のライブラリを指定します．(既定: "matplotlib")

### 設定について
`lib/config.py` 内のコメント文をご覧ください．画像増幅の設定もここにあります．

## その他
既知の問題点については、issues をご覧ください．また、問題を見つけられた場合は遠慮無く報告していただければ幸いです．  
