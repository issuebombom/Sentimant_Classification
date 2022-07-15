from utils.params_data import *
import tensorflow as tf
import tensorflow_addons as tfa # for using Rectified-Adam optimizer (instead of Adam optimizer) 
from tensorflow.keras import layers, initializers, losses, metrics

from transformers import TFBertModel


def kobert_model():

    bert_base_model = TFBertModel.from_pretrained("monologg/kobert", cache_dir='bert_ckpt', from_pt=True) 
    input_token_ids = layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_token_ids')   # tokens_tensor
    input_masks     = layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')       # masks_tensor
    input_segments  = layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_segments')    # segments_tensor  

    bert_outputs = bert_base_model([input_token_ids, input_masks, input_segments]) 
    # bert_outputs -> 0: 'last_hidden_state' & 1: 'pooler_output' (== applied GlobalAveragePooling1D on 'last_hidden_state')

    bert_outputs = bert_outputs[1] # ('pooler_output', <KerasTensor: shape=(None, 768) dtype=float32 (created by layer 'tf_bert_model')>)
    bert_outputs = layers.Dropout(0.3)(bert_outputs)
    final_output = layers.Dense(units=2, activation='softmax', kernel_initializer=initializers.TruncatedNormal(stddev=0.02), name="classifier")(bert_outputs)

    model = tf.keras.Model(inputs=[input_token_ids, input_masks, input_segments], 
                        outputs=final_output)

    # RAdam (Rectified-Adam) @ http://j.mp/2P5OmF3 / http://j.mp/2qzRjUa / http://j.mp/2N322hu / http://j.mp/2MYtPQ2
    model.compile(optimizer=tfa.optimizers.RectifiedAdam(learning_rate=1e-5, weight_decay=0.0025, warmup_proportion=0.05),
                  loss=losses.SparseCategoricalCrossentropy(), 
                  metrics=[metrics.SparseCategoricalAccuracy()])
    
    return model