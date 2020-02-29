"""
AlexNet 用設定モジュールです。
"""

from imgaug import augmenters as iaa

class DefaultConfig():
	"""
	AlexNet 用初期設定です。
	適宜インスタンス化したものを調整してください。
	"""

	def __init__(self):
		"""
		コンストラクタ。設定を実体化します。

		@param None (None): [None] この関数は引数がありません。
		@return None (None): この関数は戻り値がありません。
		"""

		# 入力画像の解像度
		# 通常変更する必要はありません。
		self.RESOLUTION = 224

		# 学習率
		self.LEARNING_RATE = 1e-3

		# 学習率を 1/10 に下げるタイミング
		# 全体のエポック数に対する割合を指定してください。1 を指定すると無効化します。
		self.TWEAK_START = 0.8

		# オプティマイザ
		# Adam か SGD から好きなのを選んでください。多分 Adam で OK です。
		self.OPTIMIZER = "Adam"

		# SGD の慣性項
		self.MOMENTUM = 0.9

		# Nesterov 加速勾配法を使用するか
		self.USE_NESTEROV = True

		# バッチサイズ
		self.BATCH_SIZE = 32

		# クラス分類
		# 分類すべきクラスを設定します。このままだとエラーになるため、データセットに合わせて必ず再設定してください。
		self.CLASSES = None

		# ログとチェックポイントの出力先
		self.LOGDIR = "./logs"

		# 処理を複数のスレッドに分割して高速化します
		self.NUM_WORKERS = 12

		# 過学習防止のための水増し
		# データセットによっては逆効果となる恐れがあるため、実際の入力を見て確認することをおすすめします。
		self.AUGMENTERS = iaa.SomeOf(n=(1, 3), children=[
			iaa.Fliplr(1), # 左右反転
			iaa.Flipud(1), # 上下反転
			iaa.Multiply((0.1, 2), per_channel=0.3), # 明るさ変更
			iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, mode="ALL", cval=(0, 255)), # 平行移動
			iaa.Affine(rotate=(-45, 45), mode="ALL", cval=(0, 255)), # 回転
			iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, mode="ALL", cval=(0, 255)), # 拡縮
			iaa.Affine(shear=(-30, 30), mode="ALL", cval=(0, 255)), # せん断
		], random_order=True)
