import os
import re
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from functions import read_fasta_sequences, one_hot_encoding

def read_numbers_from_txt(filename):
    """
    检查输入txt文件是否每行只有一个数字，若是则读取所有数字为一个列表返回。
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    if not filename.lower().endswith(".txt"):
        raise ValueError("文件扩展名必须是 .txt")

    number_pattern = re.compile(r'^[-+]?\d*\.?\d+(e[-+]?\d+)?$', re.IGNORECASE)

    numbers = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not number_pattern.match(line):
                raise ValueError(f"第 {i} 行格式错误：必须是单个数字，但得到 '{line}'")
            numbers.append(float(line))

    return numbers


def main():
    x_file_name = input("请输入FASTA格式序列文件路径（.fasta 或 .fa）: ").strip()
    y_file_name = input("请输入TXT格式标签文件路径（.txt，每行一个数字）: ").strip()

    try:
        sequences = read_fasta_sequences(x_file_name)
        y = read_numbers_from_txt(y_file_name)

        if len(sequences) != len(y):
            raise ValueError(f"序列数量 ({len(sequences)}) 与标签数量 ({len(y)}) 不一致。")

        x = [one_hot_encoding(seq) for seq in sequences]

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(x), np.array(y), test_size=0.2, random_state=42
        )
        
        print("数据加载与划分成功！")
        print(f"x_train 形状: {x_train.shape}")
        print(f"x_test 形状: {x_test.shape}")
        print(f"y_train 形状: {y_train.shape}")
        print(f"y_test 形状: {y_test.shape}")

        np.savetxt('train_data/x_train.txt', x_train.reshape(-1, 4), fmt='%.6f')
        np.savetxt('train_data/x_test.txt', x_test.reshape(-1, 4), fmt='%.6f')
        np.savetxt('train_data/y_train.txt', y_train, fmt='%.6f')
        np.savetxt('train_data/y_test.txt', y_test, fmt='%.6f')
    
    except Exception as e:
        print(f"[错误] {e}")

if __name__ == "__main__":
    main()
