import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras as keras
from tensorflow.keras.utils import get_custom_objects
from functions import one_hot_encoding, load_model, read_fasta_sequences
import random

input_name = input("请输入FASTA文件名（例如 test.fasta）: ").strip()
model_path = input("请输入模型路径（例如 model/g4_transformer.keras）: ").strip()


def extend_zimu(sequence, length=150):
    extension = length - len(sequence)
    if extension <= 0:
        return sequence
    left_len = extension // 2
    right_len = extension - left_len
    extension_seq = 'N' * extension
    extended_sequence = extension_seq[:left_len] + sequence + extension_seq[left_len:]
    return extended_sequence


sequences = read_fasta_sequences(input_name)
out_seq = []
for seq in sequences:
    center = seq[75:125]  # 提取中心区域
    padded_seq = extend_zimu(center, length=150)
    onehot = one_hot_encoding(padded_seq)
    out_seq.append(onehot)

print("加载模型中...")
model = load_model(model_path)
print("模型加载完成，开始预测...")

X = np.array(out_seq)
score = model.predict(X)

output_file = "result/result.txt"
np.savetxt(output_file, score, fmt='%.6f')

print(f"预测完成，结果已保存至 {output_file}")
