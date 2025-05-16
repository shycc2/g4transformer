import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import get_custom_objects

def read_fasta_sequences(filename):
    """
    检查输入文件是否为FASTA格式，并读取其中的所有序列（假设每条记录由两行组成：一行标签，一行序列）。
    返回所有序列组成的列表（只返回序列，不含标签）。
    """
    import os

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    if not filename.lower().endswith(('.fasta', '.fa')):
        raise ValueError("输入文件不是FASTA格式（必须以 .fasta 或 .fa 结尾）")
    
    try:
        sequences = []
        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]  # 去掉空行和换行
            if len(lines) % 2 != 0:
                raise ValueError("FASTA 文件格式错误：标签行与序列行数不匹配。")

            for i in range(0, len(lines), 2):
                header = lines[i]
                sequence = lines[i+1]
                if not header.startswith(">"):
                    raise ValueError(f"第 {i+1} 行不是合法的FASTA标签行：{header}")
                sequences.append(sequence)
        
        if not sequences:
            raise ValueError("FASTA文件中没有找到任何序列。")
        
        return sequences
    
    except Exception as e:
        raise ValueError(f"读取FASTA文件失败: {e}")

def one_hot_encoding(seq):
    mapping = {'A': [1, 0, 0, 0],
               'G': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'T': [0, 0, 0, 1]}
    return [mapping.get(base, [0.25, 0.25, 0.25, 0.25]) for base in seq]

def load_model(model_name):
    if model_name != 'model/g4_transformer.keras' and model_name != 'model/g4_transformer_attention.keras':
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
    else:
        class PositionalEncoding(layers.Layer):
            def __init__(self, max_len=150, d_model=128):
                super().__init__()
                self.pos_emb = self.add_weight(
                    "pos_emb",
                    shape=(1, max_len, d_model),
                    initializer="glorot_normal"
                )

            def call(self, x):
                return x + self.pos_emb
                
    get_custom_objects().update({'PositionalEncoding': PositionalEncoding})
    model = tf.keras.models.load_model(model_name)
    return model