"""
AlexNet メインモジュールです。
"""

import os

import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras

# TODO
assert keras.backend.image_data_format() == 'channels_last', "すまん"

REVISION = 1


class AlexNet(object):
	"""
	突貫工事の AlexNet
	
	Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton: ImageNet Classification with Deep Convolutional Neural Networks
	https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
	"""

	def __init__(self, config=None):
		"""
		AlexNet インスタンスを初期化します。

		@param config (DefaultConfig): [必須] 設定のインスタンス。
		@return None (None): この関数は戻り値がありません。
		"""

		assert config, "設定を指定してください。"
		assert config.CLASSES, "分類すべきクラスが未設定です。"

		self.config = config
		self.model = self.build()

		# 設定を表示します
		for key, value in self.config.__dict__.items():
			print("%s\t = %s" % (key.ljust(20), value))
		print("_________________________________________________________________")


	def build(self):
		"""
		AlexNet を生成します。

		@param None (None): [None] この関数は引数がありません。
		@return model (keras.models.Model): Keras のモデル。
		"""

		# メモリ節約
		# 何も指定しないと TensorFlow は利用可能な全メモリを専有しようとします。
		# このオプションにより必要最小限のメモリを使うようになります。
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		session = tf.Session(config=config)
		keras.backend.set_session(session)

		# オプティマイザ
		# オプティマイザは損失関数が小さくなる方向にモデルを導きます。
		opt = self.config.OPTIMIZER
		lr = self.config.LEARNING_RATE # パラメータ更新の大きさ
		if opt == "SGD":
			# SGD; Stochastic Gradient Decent （確率的勾配降下法）は古典的なオプティマイザです。
			optimizer = keras.optimizers.SGD(lr=lr, momentum=self.config.MOMENTUM, nesterov=self.config.USE_NESTEROV)
		elif opt == "Adam":
			# Adam は SGD と RMSProp を組み合わせた収束の速いオプティマイザです。
			# Nadam は Nesterov Accelerated Gradient (NAG) と RMSProp を組み合わせたオプティマイザです。
			# 本実装では設定を変えない限り Nadam が使われます。
			if self.config.USE_NESTEROV:
				optimizer = keras.optimizers.Nadam(lr=lr)
			else:
				optimizer = keras.optimizers.Adam(lr=lr)
		else:
			raise ValueError("オプティマイザが変です。")

		# 損失関数
		# 損失関数は現在のモデルの状態と理想的なモデルの状態との距離を現します。
		# 損失関数が小さくなるように訓練が進みます。
		if len(self.config.CLASSES) > 2:
			loss = 'categorical_crossentropy' # 多値分類
		else:
			loss = 'binary_crossentropy' # 二値分類

		# 入力解像度
		res = self.config.RESOLUTION

		# Sequential モデルインスタンス（層を重ねるだけでネットワークができます）を初期化します。
		# AlexNet は5つの畳み込み層と3つの全結合層から構成されます。
		# 畳み込みにより入力画像がもつ特徴（浅い層では縦線や横線、深い層では目や耳、顔など）が抽出されていきます。
		# 全結合層は畳み込み層が出力した特徴量を基に判断を下します。
		model = keras.models.Sequential([
			# Stage 1. 入力層 1回目の畳み込み
			# 96 は層が生成する特徴の種類です。11は一度にどのくらいの範囲（ピクセル数）を見るかです。
			keras.layers.Conv2D(96, 11, strides=4, bias_initializer='zeros', input_shape=(res, res, 3)),

			# このままだと解像度が大きすぎるため特徴マップを小さくします。
			keras.layers.MaxPooling2D(pool_size=3, strides=2),

			# Batch Normalization は過学習（特定のデータに適合しすぎること）を防いだり精度を上げたりする効果があります。
			# 本実装では本家 AlexNet の Local Response Normalization の代わりに用いています。
			keras.layers.BatchNormalization(),

			# モデルに非線形性をもたせます。
			# 非線形にすることでより複雑な関数を近似することができます。
			keras.layers.ReLU(),

			# Stage 2. 隠れ層 2回目の畳み込み
			keras.layers.Conv2D(256, 5, bias_initializer='ones'),
			keras.layers.MaxPooling2D(pool_size=3, strides=2),
			keras.layers.BatchNormalization(),
			keras.layers.ReLU(),

			# Stage 3. 隠れ層 3回目の畳み込み
			keras.layers.Conv2D(384, 3, activation='relu', bias_initializer='zeros'),

			# Stage 4. 隠れ層 4回目の畳み込み
			keras.layers.Conv2D(384, 3, activation='relu', bias_initializer='ones'),

			# Stage 5. 隠れ層 5回目の畳み込み
			keras.layers.Conv2D(256, 3, bias_initializer='ones'),
			keras.layers.MaxPooling2D(pool_size=3, strides=2),
			keras.layers.BatchNormalization(),
			keras.layers.ReLU(),

			# Stage 6. 隠れ層 1回目の全結合層
			# このままだとテンソルの形が異なるため、全結合層で処理できるように変形します。
			keras.layers.Flatten(),

			# これは4096個のニューロンが前後の層と互いに接続された全結合層です。
			keras.layers.Dense(4096, activation='relu'),

			# ニューロンの多い全結合層はしばしば過学習を起こします。
			# 過学習は特定のニューロンに依存しすぎることで起きると言われています。
			# それを防ぐために50%の確率でランダムにニューロンを無かったことにします。
			keras.layers.Dropout(0.5),

			# Stage 7. 隠れ層 2回目の全結合層
			keras.layers.Dense(4096, activation='relu'),
			keras.layers.Dropout(0.5),

			# Stage 8. 出力層 3回目の全結合層
			# 分類すべきクラス数分のニューロンを用意します。
			# 最後の softmax 関数による活性化でどのニューロンがもっとも強く反応しているかを求めます。
			keras.layers.Dense(len(self.config.CLASSES), activation='softmax')
		], name="AlexNet")

		model.summary()
		model.compile(loss=loss, metrics=['accuracy'], optimizer=optimizer)

		return model


	def train(self, train, val, epochs=10, initial_epoch=0, class_weight=None):
		"""
		モデルの訓練を行います。

		@param train (keras.utils.Sequential): [必須] 訓練用データセットジェネレーター。
		@param val (keras.utils.Sequential): [任意] 確認用データセットジェネレーター。
		@param epochs (int): [任意] 何エポック回すか。
		@param initial_epoch (int): [任意] ここから再開したことにします（学習を中断したときに便利です）。
		@param class_weights (dict): [任意] 不均衡データの場合、どのクラスにどれだけ重みづけするかを設定します。
		@return history (よくわからないもの): 訓練の履歴。中止した場合は何も返りません。
		"""

		print("学習を開始します。")

		try:
			history = self.model.fit_generator(
				train,
				validation_data=val,
				workers=self.config.NUM_WORKERS,
				epochs=epochs,
				initial_epoch=initial_epoch,
				callbacks=[
					keras.callbacks.TerminateOnNaN(),
					keras.callbacks.ModelCheckpoint(
						os.path.join(self.config.LOGDIR, "alex-{epoch:04d}.h5"),
						save_best_only=True,
						save_weights_only=True,
						verbose=True,
						monitor='val_loss' if val is not None else 'loss'
					),
					keras.callbacks.LearningRateScheduler(
						lambda x: self.config.LEARNING_RATE / 10 if x >= epochs * self.config.TWEAK_START else self.config.LEARNING_RATE,
						verbose=1
					),
					keras.callbacks.TensorBoard(log_dir=self.config.LOGDIR)
				],
				class_weight=class_weight
			)
		except KeyboardInterrupt:
			print("訓練中止")
			return
		except tf.errors.ResourceExhaustedError:
			print("AlexNet の ぼうけん（損失関数の山下り）は これで おわってしまった！！")
			print("死因: メモリが足りません。")
			print("モデルの状態を退避しています...")
			self.model.save_weights(os.path.join(self.config.LOGDIR, "alex-ree.h5"))
			raise

		return history


	def restore(self, filename):
		"""
		チェックポイントからモデルの状態を復元します。

		@param filename (str): [必須] チェックポイントのファイル名
		@return None (None): この関数は戻り値がありません。
		"""

		path = os.path.join(self.config.LOGDIR, filename)

		print("%s からモデルを復元します。" % path)
		self.model.load_weights(path, by_name=True)


	def predict(self, image):
		"""
		推論を行います。

		@param image ([height, width, 3], dtype=np.uint8): [必須] OpenCVなどで読み込んだRGB画像。
		@return pred ([len(self.config.CLASSES)], dtype=np.float?): 各クラスの尤度。
		"""

		# 設定
		res = self.config.RESOLUTION

		# 入れ物
		ph_data = np.zeros([1, res, res, 3], dtype=np.float16)

		# ネットワークの形に合わせます
		if image.shape[0:2] != [res, res]:
			image = cv2.resize(image, (res, res), cv2.INTER_CUBIC)

		# 入れ物に入れます
		ph_data[0] = image / 255.

		pred = self.model.predict(ph_data)

		return pred[0]
