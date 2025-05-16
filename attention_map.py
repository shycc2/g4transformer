import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.initializers import GlorotNormal
import seaborn as sns
import os
from functions import one_hot_encoding, load_model


# 用户输入模型路径和序列
model_path = input("请输入模型路径（例如 model/g4_transformer_attention.keras）: ").strip()
print("使用model/g4_trnasformer_attention时请保持序列长度为150nt")
sequence = input("请输入DNA序列（A/C/G/T字符）: ").strip().upper()

assert set(sequence).issubset({'A', 'C', 'G', 'T'}), "序列中只能包含 A, C, G, T"

# 加载模型
model = load_model(model_path)

# One-hot 编码输入
fi = np.expand_dims(one_hot_encoding(sequence), axis=0)

# 获取注意力权重输出
attn_weights_list = model.predict(fi)

# 创建输出文件夹
os.makedirs("pics/attention_map", exist_ok=True)

# 可视化每一层的注意力图
for layer_idx, attn_weights in enumerate(attn_weights_list):
    merged_attention = tf.reduce_mean(attn_weights, axis=1)  # 平均合并多头注意力
    merged_attention = merged_attention.numpy()[0]

    plt.figure(figsize=(10, 8))
    sns.heatmap(merged_attention, cmap='viridis', annot=False)
    plt.title(f"Merged Attention Map (Layer {layer_idx+1})")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    filename = f"pics/attention_map/attention_general_{layer_idx+1}.png"
    plt.savefig(filename)
    plt.show()
    print(f"已保存 Layer {layer_idx+1} 的注意力图至: {filename}")
