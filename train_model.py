import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import GlorotNormal

# 自定义位置编码层
class PositionalEncoding(layers.Layer):
    def __init__(self, length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.length = length
        self.d_model = d_model
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.length, self.d_model),
            initializer=GlorotNormal()
        )

    def call(self, x):
        return x + self.pos_emb[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"length": self.length, "d_model": self.d_model})
        return config

# Transformer 编码器模块
def transformer_encoder(inputs, d_model, nhead, dim_feedforward, dropout, layer_idx=0):
    attn_output, attn_scores = layers.MultiHeadAttention(
        num_heads=nhead,
        key_dim=d_model // nhead,
        name=f"transformer_{layer_idx}_attn"
    )(inputs, inputs, return_attention_scores=True)

    attn_output = layers.Dropout(dropout)(attn_output)
    attn_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = layers.Dense(dim_feedforward, activation="relu")(attn_output)
    ffn = layers.Dense(d_model)(ffn)
    ffn = layers.Dropout(dropout)(ffn)

    return layers.LayerNormalization(epsilon=1e-6)(attn_output + ffn), attn_scores

# 构建 Transformer 模型
def build_transformer_model(length, d_model, layer_num, nhead, dim_feedforward, dropout):
    inputs = layers.Input(shape=(length, 4))
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(length, d_model)(x)

    all_attention_scores = []
    for i in range(layer_num):
        x, attn_scores = transformer_encoder(x, d_model, nhead, dim_feedforward, dropout, i)
        all_attention_scores.append(attn_scores)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="classifier")(x)

    train_model = Model(inputs, outputs)
    attn_model = Model(inputs, all_attention_scores)
    return train_model, attn_model

# 从参数CSV读取训练参数
def read_parameter(file_name):
    df = pd.read_csv(file_name)
    return (
        int(df['d_model'].iloc[0]),
        int(df['layer_num'].iloc[0]),
        int(df['nhead'].iloc[0]),
        int(df['dim_feedforward'].iloc[0]),
        float(df['dropout'].iloc[0]),
        float(df['learning_rate'].iloc[0]),
        int(df['epochs'].iloc[0]),
        int(df['batch_size'].iloc[0]),
        float(df['validation_split'].iloc[0])
    )

# 主程序
def main():
    print("=== Transformer 模型训练脚本 ===")
    x_train_file = input("请输入 x_train 数据文件名（如 x_train.txt）: ").strip()
    y_train_file = input("请输入 y_train 数据文件名（如 y_train.txt）: ").strip()
    param_file = input("请输入参数文件名（如 parameter.csv）: ").strip()

    x_path = os.path.join("train_data", x_train_file)
    y_path = os.path.join("train_data", y_train_file)

    if not os.path.isfile(x_path):
        print(f"[错误] 找不到 x_train 文件：{x_path}")
        return
    if not os.path.isfile(y_path):
        print(f"[错误] 找不到 y_train 文件：{y_path}")
        return
    if not os.path.isfile(param_file):
        print(f"[错误] 找不到参数文件：{param_file}")
        return

    print("加载数据中...")
    x_train = np.loadtxt(x_path).reshape(-1, 150, 4)
    y_train = np.loadtxt(y_path).reshape(-1, 1)

    print("读取超参数中...")
    d_model, layer_num, nhead, dim_feedforward, dropout, learning_rate, epochs, batch_size, validation_split = read_parameter(param_file)

    length = x_train.shape[1]

    print("构建模型中...")
    train_model, attn_model = build_transformer_model(length, d_model, layer_num, nhead, dim_feedforward, dropout)

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("开始训练...")
    history = train_model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split
    )

    os.makedirs("model", exist_ok=True)
    train_model.save("model/model.keras")
    attn_model.save("model/attention.keras")
    print("模型已保存到 model/model.keras 和 model/attention.keras")

if __name__ == "__main__":
    main()
