import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras as keras
from transformers.optimization_tf import WarmUp
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.utils import get_custom_objects
from alibi.explainers import IntegratedGradients
from deeplift.visualization import viz_sequence
from functions import one_hot_encoding, load_model

model_path = input("请输入模型路径（例如 model/model.keras）：").strip()
print("使用model/g4_trnasformer_attention时请保持序列长度为150nt")
sequence = input("请输入一个DNA序列：（A/C/G/T字符）").strip().upper()
assert all(c in "ACGT" for c in sequence), "序列只能包含 A, C, G, T"

fi = np.expand_dims(one_hot_encoding(sequence), axis=0)

model = load_model(model_path)

class AutoModelWrapper(keras.Model):
    def __init__(self, transformer: keras.Model, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        transformer
            Transformer to be wrapped.
        """
        super().__init__()
        self.transformer = transformer

    def call(self,
             input_ids: Union[np.ndarray, tf.Tensor],
             attention_mask: Optional[Union[np.ndarray, tf.Tensor]] = None,
             training: bool = False):
        """
        Performs forward pass throguh the model.


        Parameters
        ----------
        input_ids
            Indices of input sequence tokens in the vocabulary.
        attention_mask
            Mask to avoid performing attention on padding token indices.

        Returns
        -------
        Classification probabilities.
        """
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask, training=training)
        return tf.nn.softmax(out.logits, axis=-1)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
fi = np.expand_dims(one_hot_encoding(sequence), axis=0)
auto_model = AutoModelWrapper(model)
layer = model.layers[2]
n_steps = 50
method = "gausslegendre"
internal_batch_size = 100
nb_samples = 10
ig  = IntegratedGradients(model,
                          layer=layer,
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)

predictions = model(fi).numpy().argmax(axis=1)
explanation = ig.explain(fi, baselines=None, target=predictions, attribute_to_layer_inputs=False)
attrs = explanation.attributions[0].sum(axis=2)

letter_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
length = len(sequence)
array_4 = np.zeros((len(sequence), 4))
letter_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

for i, letter in enumerate(sequence):
    col_index = letter_to_index[letter]
    array_4[i, col_index] = attrs[0,i]  

def plot_a(ax, base, left_edge, height, color):
    a_polygon_coords = [
        np.array([
           [0.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.2, 0.0],
        ]),
        np.array([
           [1.0, 0.0],
           [0.5, 1.0],
           [0.5, 0.8],
           [0.8, 0.0],
        ]),
        np.array([
           [0.225, 0.45],
           [0.775, 0.45],
           [0.85, 0.3],
           [0.15, 0.3],
        ])
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords
                                                 + np.array([left_edge,base])[None,:]),
                                                facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,
                                            facecolor=color, edgecolor=color))
    ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,
                                            facecolor='white', edgecolor='white'))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,
                                            facecolor='white', edgecolor='white', fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,
                                            facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,
                                            facecolor=color, edgecolor=color, fill=True))


def plot_t(ax, base, left_edge, height, color):
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],
                  width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
    ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],
                  width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))

default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,
                 height_padding_factor,
                 length_padding,
                 subticks_frequency,
                 highlight,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs):
    if len(array.shape)==3:
        array = np.squeeze(array)
    assert len(array.shape)==2, array.shape
    if (array.shape[0]==4 and array.shape[1] != 4):
        array = array.transpose(1,0)
    assert array.shape[1]==4
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):

        acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color=colors[letter[0]]
            if (letter[1] > 0):
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]                
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color)
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    #now highlight any desired positions; the key of
    #the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= array.shape[0]
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(xy=[start_pos,min_depth],
                    width=end_pos-start_pos,
                    height=max_height-min_depth,
                    edgecolor=color, fill=False))
            
    ax.set_xlim(-length_padding, array.shape[0]+length_padding)
    #ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
    height_padding = max(abs(min_neg_height)*(height_padding_factor),
                         abs(max_pos_height)*(height_padding_factor))
    ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
    ax.set_xticks([])        
    ax.set_xticklabels([])   
    ax.set_ylabel('Attibution Score', fontsize=12)


def plot_weights(array,
                 name,
                 figsize=(20,2),
                 height_padding_factor=0.2,
                 length_padding=1.0,
                 subticks_frequency=1.0,
                 colors=default_colors,
                 plot_funcs=default_plot_funcs,
                 highlight={}
                 ):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111) 
    plot_weights_given_ax(ax=ax, array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    
plot_weights(array_4,
             name='pics/logo/logo.png',
             figsize=(12, 2),
             height_padding_factor=0.2,
             subticks_frequency=25)

print("绘图完成，已保存至pics/logo/logo.png")
