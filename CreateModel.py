import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import config
from scipy.interpolate import interp1d
import data_processing as dp
from functools import partial
import asyncio


async def train_and_evaluate_model_async(base_data_path):
    loop = asyncio.get_running_loop()
    # CPUバウンドタスクの実行には、デフォルトのexecutorをそのまま使用
    model = await loop.run_in_executor(
        None,  # Noneはデフォルトのexecutorを使用することを意味します
        partial(train_and_evaluate_model_for_finetuning, base_data_path)  # 引数を持つ関数のためのpartial適用
    )
    return model

def train_and_evaluate_model_for_finetuning(base_data_path):
    dp.check_csv_data(base_data_path)

    # BaseData.csvを読み込む
    base_data = pd.read_csv(base_data_path)

    # ノイズデータとタイムワーピングデータの生成
    noise_data = dp.generate_noise_data(base_data, 500)
    time_warped_data = dp.generate_time_warped_data(base_data)

    coloumn_names = [f'col{i}' for i in range(1, base_data.shape[1] + 1)]

    base_data.columns = coloumn_names
    noise_data.columns = coloumn_names
    time_warped_data.columns = coloumn_names

    # データの結合
    combined_data = pd.concat([base_data, noise_data, time_warped_data], ignore_index=True)

    # 入力データとラベルの分離
    x_data = combined_data.iloc[:, :-config.num_cols].values
    y_data = combined_data.iloc[:, -config.num_cols:].values

    # 事前に訓練されたモデルをロード
    model_path = config.parent_dir + "/advanced_model"
    existing_model = keras.models.load_model(model_path)

    # 事前に訓練されたモデルの最終層を削除し、新しい出力層を追加
    existing_model.layers.pop()  # 最終層を削除
    new_output_layer = keras.layers.Dense(config.num_cols, activation='softmax', name='new_output_layer')(existing_model.layers[-2].output)
    new_model = keras.models.Model(inputs=existing_model.input, outputs=new_output_layer)

    # モデルのコンパイル
    new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 新しいデータでモデルを再訓練
    new_model.fit(x_data, y_data, epochs=10, batch_size=10, validation_split=0.2)

    return new_model