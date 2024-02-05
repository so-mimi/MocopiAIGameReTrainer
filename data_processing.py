import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import os

import config

# タイムワーピング関数
def time_warping(data, alpha=0.01):
    scale = 1.0 + np.random.uniform(-alpha, alpha)
    old_times = np.linspace(0, 1, data.shape[0])
    new_times = np.linspace(0, 1, round(data.shape[0] * scale))
    interpolator = interp1d(old_times, data, axis=0, fill_value='extrapolate')
    new_data = interpolator(new_times)
    
    if new_data.shape[0] < data.shape[0]:
        pad_length = data.shape[0] - new_data.shape[0]
        new_data = np.vstack([np.zeros((pad_length, data.shape[1])), new_data])
    else:
        new_data = new_data[:data.shape[0]]
    return new_data

def generate_noise_data(base_data, num_samples=500, noise_strength_range=(0, 0.0025)):
    num_rows, num_cols = base_data.shape

    # ランダムインデックスを生成
    random_row_indices = np.random.randint(0, num_rows, size=num_samples)
    
    # 選択された行に対してノイズを追加
    selected_rows = base_data.iloc[random_row_indices].values  # Pandas DataFrameからNumPy配列へ変換

    # ノイズ強度をランダムに選択し、それを用いてノイズを生成
    noise_strengths = np.random.uniform(noise_strength_range[0], noise_strength_range[1], size=(num_samples, 1))
    noises = noise_strengths * np.random.randn(num_samples, num_cols)

    # ノイズを追加
    noised_data = np.clip(selected_rows + noises, -1, 1)

    return pd.DataFrame(noised_data, columns=base_data.columns)

def check_csv_data(file_path):
    # CSVファイルを読み込む
    data = pd.read_csv(file_path)

    # 行数と列数を取得
    num_rows, num_cols = data.shape
    print(f"行数: {num_rows}, 列数: {num_cols}")
    config.num_cols = num_cols - 2610

    
    # すべての行の列数が一致しているか確認
    for index, row in data.iterrows():
        if len(row) != num_cols:
            raise ValueError(f"行 {index} の列数が一致しません。")

    print("全ての行の長さが一致しています。")
    

def generate_onnx_model(base_data_path):

    check_csv_data(base_data_path)

    # BaseData.csvを読み込む
    base_data = pd.read_csv(base_data_path)

    # ノイズデータの生成とタイムワーピングデータの生成をメモリ上で行う
    noise_data_list = []
    time_data_list = []

    # ノイズデータの生成を呼び出し
    noise_data = generate_noise_data(base_data, 500)
    
    for _ in range(600):  # タイムワーピングデータの生成
        num_rows = base_data.shape[0]
        random_row_index = np.random.randint(num_rows)
        selected_row = base_data.iloc[random_row_index]
        selected_data = selected_row.values.reshape(-1, 87)  # ここはデータ形状に応じて調整が必要かもしれません
        warped_data = time_warping(selected_data).reshape(-1)
        time_data_list.append(warped_data)

    # データフレームに変換
    noise_data = pd.DataFrame(noise_data_list, columns=base_data.columns)
    time_data = pd.DataFrame(time_data_list, columns=base_data.columns[:-config.num_cols] + ['label'] * config.num_cols)  # 'label'は適宜調整

    # データの結合
    combined_data = pd.concat([base_data, noise_data, time_data], ignore_index=True)

    x_finetune = combined_data.iloc[:, :-config.num_cols].values
    y_finetune = combined_data.iloc[:, -config.num_cols:].values

    # モデルの訓練とONNX形式での保存を行う部分は、既存のコードを使用できます。

    # 既存のモデルを読み込む
    config.parent_dir = os.path.dirname(base_data_path)
    model_path = config.parent_dir + "/advanced_model"
    existing_model = keras.models.load_model(model_path)

    # 既存のモデルに新しい出力層を追加
    new_output_layer = keras.layers.Dense(config.num_cols, activation='softmax', name='new_output')(existing_model.layers[-2].output)

    # 新しいモデルを構築
    new_model = keras.models.Model(inputs=existing_model.input, outputs=new_output_layer)

    # 既存のモデルから新しいモデルへの重みのコピー
    for layer in new_model.layers[:-1]:
        if layer.name in existing_model.layers:
            layer.set_weights(existing_model.get_layer(layer.name).get_weights())

    # モデルのコンパイル
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 新しいデータでモデルを再訓練
    new_model.fit(x_finetune, y_finetune, epochs=10, batch_size=10, validation_split=0.2)

    # TensorFlowのモデルをONNX形式に変換
    spec, _ = tf2onnx.convert.from_keras(new_model, opset=13)

    onnx_file_path = config.parent_dir + "/mocopiAI.onnx"

    # 出力ファイルパスにONNXモデルを保存
    onnx.save_model(spec, onnx_file_path)