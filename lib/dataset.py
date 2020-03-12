"""
AlexNet データセットモジュールです。
"""

import os
import random

import numpy as np
import cv2

from tensorflow import keras

class DefaultDataGenerator(keras.utils.Sequence):
	"""
	MNIST や CIFAR-10 の訓練および確認データセットを生成します。
	"""

	def __init__(self, config=None, what="mnist", mode="train"):
		"""
		MNIST または CIFAR-10 データセットのインスタンスを生成します。

		@param config (DefaultConfig): [必須] 設定
		@param what (str): [任意] mnist または cifar10
		@param mode (str): [任意] train で訓練用データセットを、val で確認用データセットを生成します。
		@return None (None): この関数は戻り値がありません。
		"""

		self.config = config
		self.mode = mode
		self.what = what
		self.batch_size = config.BATCH_SIZE

		if what == "mnist":
			print("MNIST データを読み込んでいます...")
			(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
			self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
		elif what == "cifar10":
			print("CIFAR-10 データを読み込んでいます...")
			(train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
			self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
		else:
			raise ValueError("対応していないデータセットです。")

		if self.mode == "train":
			self.data_x = train_x
			self.data_y = train_y
			del test_x
			del test_y
		elif self.mode == "val":
			self.data_x = test_x
			self.data_y = test_y
			del train_x
			del train_y
		else:
			raise ValueError("Generator のモードが変です。")

		# データセットが訓練用以外の場合、必ず割り切れるようにバッチサイズを再設定します。
		# 確認用データセットでは、オーバーラップによる結果の不確定性が許容されません。
		# バッチサイズを1にしてしまえば話は早いのですが、バリデーション時の Keras のプログレスバーがブワアーーってなります。
		if self.mode != "train":
			length = len(self.data_x)
			while self.batch_size > 1:
				if length % self.batch_size == 0:
					break
				else:
					self.batch_size -= 1
			print("バッチサイズが %d に再設定されました。" % self.batch_size)


	def __getitem__(self, idx):
		"""
		1バッチを返します。

		@param idx (int): [必須] 0から始まるバッチ番号。
		@return x (np.float): [バッチサイズ, 解像度, 解像度, 3] の MNIST 手書き文字画像または CIFAR-10 画像。
		@return y (np.bool): One-hot エンコーディングされたラベル。
		"""

		# 長いので略称をつけます。
		bs = self.batch_size
		res = self.config.RESOLUTION

		# 入れ物を作ります。
		x = np.zeros([bs, res, res, 3], dtype=np.uint8)
		y = np.zeros([bs, len(self.classes)], dtype=np.bool)

		for i in range(bs):
			# サンプル番号を計算します。
			# 余りの計算はデータセットがバッチサイズで割り切れなかったときのためです。
			index = (idx * bs + i) % len(self.data_x)
			temp = self.data_x[index]

			if self.what == "mnist":
				temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

			temp = cv2.resize(temp, (res, res))

			x[i] = temp
			y[i] = keras.utils.to_categorical(self.data_y[index], num_classes=len(self.classes))

		# 過学習を防ぐための水増し。
		if self.mode == "train":
			x = self.config.AUGMENTERS.augment_images(x)

		# ネットワークの入力に適した状態にします（正規化）。
		x = x / 255.0

		return x, y


	def __len__(self):
		"""
		1エポックを構成するバッチの個数を返します。

		@param None (None): [None] この関数は引数がありません。
		@return (int): 1エポックを構成するバッチの個数
		"""

		return int(np.ceil(self.data_x.shape[0] / self.batch_size))


class DirectoryBasedDataGenerator(keras.utils.Sequence):
	"""
	ディレクトリベースのデータジェネレーターです。fit_generator() 関数に渡してご使用ください。
	フォルダを作って画像を放り込むだけでデータセットができます。
	"""

	def __init__(self, config=None, root=None, mode="train", length_multiplier=1, extensions=(".jpg", ".png")):
		"""
		データセットをインスタンス化します。

		@param config (DefaultConfig): [必須] 設定
		@param root (str): [必須] 走査するディレクトリの根っこ。
		@param mode (str): [任意] train で訓練用データセットを、val で確認用データセットを生成します。
		@param length_multiplier (int): [任意]
			小規模なデータセットを用いる際のエポック開始時のオーバーヘッドを削減するため、keras に報告するデータセットの長さを 本来の長さ×length_multiplier に偽装します。
			実験的機能です。精度に影響を及ぼす可能性がありますのでご注意ください。
		@param extensions ((str)): [任意] 読み込む画像形式のタプル。OpenCV が対応している拡張子のみ指定できます。
		@return None (None): この関数は戻り値がありません。
		"""

		assert config, "設定を指定してください。"
		assert root, "走査するディレクトリのルートを指定してください。"

		if not os.path.exists(root):
			raise FileNotFoundError(root)

		self.config = config
		self.mode = mode
		self.batch_size = config.BATCH_SIZE
		self.data = [] # 各画像へのファイルパスを保持します。
		self.classes = [] # 各クラス名を保持します。
		self.class_samples = {} # 各クラスのサンプル数を保持します。
		self.class_weight = {} # 不均衡データをなんとかします（できるとは言ってない）。
		self.length_multiplier = length_multiplier
		self.extensions = extensions

		# ディレクトリ内を走査します。
		for current_dir, dirs, files in os.walk(root):
			if len(dirs) == 0:
				# 中にフォルダが無い階層（最下層）まで来たら、そのフォルダ名をクラス名とします。
				class_name = os.path.basename(current_dir)

				# classes 変数にフォルダ名を追加します。
				self.classes.append(class_name)

				# 不均衡データ対策をするためにクラスのサンプルを数えます。
				samples = 0

				# data 変数にファイルパスを追加します。
				for file in files:
					_, ext = os.path.splitext(file)
					if ext in self.extensions:
						self.data.append(os.path.join(current_dir, file))
						samples += 1

				assert samples > 0, "0 サンプルのクラスは許容されません。"

				self.class_samples[class_name] = samples

		print("%d 個のサンプルが見つかりました。" % len(self.data))

		# 再現性をもたせるために順番に並べ替えます。
		# これがないとデータセットを作るたびにクラスが変わるという悲惨なことが起きます。
		self.classes.sort()

		# 一方で data 変数はシャッフルします。
		# 1バッチ内のサンプルの偏りを減らすためです。
		random.shuffle(self.data)

		# データセットが訓練用以外の場合、必ず割り切れるようにバッチサイズを再設定します。
		# 確認および評価用データセットでは、オーバーラップによる結果の不確定性が許容されません。
		# バッチサイズを1にしてしまえば話は早いのですが、バリデーション時の Keras のプログレスバーがブワアーーってなります。
		if self.mode != "train":
			length = len(self.data)
			while self.batch_size > 1:
				if length % self.batch_size == 0:
					break
				else:
					self.batch_size -= 1
			print("バッチサイズが %d に再設定されました。" % self.batch_size)

		# 最大サンプル数を探します。
		max_samples = max(self.class_samples.values())

		# クラスごとの重みを設定します。
		for i, c in enumerate(self.classes):
			# 最大のサンプル数をもつクラスが重み1になるようにします。
			w = max_samples / self.class_samples[c]
			self.class_weight[i] = w

			print("クラス: %s(%d), 重み: %f" % (c, i, w))

	def __len__(self):
		"""
		1エポックを構成するバッチの個数を返します。

		@param None (None): [None] この関数は引数がありません。
		@return (int): 1エポックを構成するバッチの個数
		"""

		return int(np.ceil(len(self.data) / self.batch_size * self.length_multiplier))

	def __getitem__(self, idx):
		"""
		1バッチ分のデータとラベルの組を返します。

		@param idx: [必須] (int) バッチのインデックス（索引）。0, 1, 2, ...
		@return x: (np.array) 1バッチ分のデータ。
		@return y: (np.array) 1バッチ分のラベル。
		"""

		# 長いので略称をつけます。
		bs = self.batch_size
		res = self.config.RESOLUTION

		# 入れ物を作ります。
		x = np.zeros([bs, res, res, 3], dtype=np.uint8)
		y = np.zeros([bs, len(self.classes)], dtype=np.bool)

		for i in range(bs):
			# サンプル番号を計算します。
			# 余りの計算は length_multiplier > 1 のときとデータセットがバッチサイズで割り切れなかったときのためです。
			index = (idx * bs + i) % len(self.data)

			# 画像を読みます。
			data = cv2.imread(self.data[index])
			data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
			data = cv2.resize(data, (res, res), interpolation=cv2.INTER_CUBIC)

			# ラベルを読みます。
			# そのままだと文字列になっていて使えないので、classes に登録された順番（数値）に変換します。
			label = os.path.basename(os.path.dirname(self.data[index]))
			label = self.classes.index(label)

			# 入れ物に入れます。
			# ラベルは one-hot 表現に変換します。
			x[i] = data
			y[i] = keras.utils.to_categorical(label, num_classes=len(self.classes))

		# 過学習を防ぐための水増し。
		if self.mode == "train":
			x = self.config.AUGMENTERS.augment_images(x)

		# ネットワークの入力に適した状態にします（正規化）。
		x = x / 255.0

		return x, y

	def on_epoch_end(self):
		"""
		1エポックが終わった後に実行する処理を書きます。

		@param None (None): [None] この関数は引数がありません。
		@return None (None): この関数は戻り値がありません。
		"""

		random.shuffle(self.data)
