
from keras.layers import merge, Input, TimeDistributed, Dense, Lambda, K
from keras import Model
import numpy as np
from my_layers import ThreeDOut

hidden = 1
context = Input(shape=(5, 4, 2), dtype='float32', name='story')
question = Input(shape=(5, 2), dtype='float32', name='question')

joint_enc2 = TimeDistributed(ThreeDOut(2))(question)
m = merge([question, context], mode="dot", dot_axes=(2, 3))
pooled_attn = Lambda(lambda x: K.max(x, axis=3),
	output_shape=lambda shape: ((shape[0], shape[1], shape[2])))(m)


model = Model(
    inputs=[context, question],
    outputs=[pooled_attn]
)

print(model.summary())