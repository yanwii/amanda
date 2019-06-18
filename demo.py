import sys
import os
import numpy

from keras.layers.convolutional import *
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import *
from keras.layers import recurrent, Input, merge, Masking
from keras.layers.recurrent import SimpleRNN
from keras.layers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from my_layers import *
from config import args


class QAModel(object):
	def __init__(self,):
		self.embed_mat = args.embed_mat
		self.embed_dim = self.embed_mat.shape[1]
		self.vocab_size = self.embed_mat.shape[0]
		self.char_embedding = args.char_embedding # True or False
		if self.char_embedding:
			self.char_embed_dim = args.char_embed_dim
			self.char_vocab_size = args.char_vocab_size
			self.char_cnn_filter_width = args.char_cnn_filter_width
		self.qtype = args.qtype   # 'wh2' or 'first2'
		self.h = args.hidden_dim
		self.maxwordlen = args.maxwordlen
		if args.border_mode is not None:
			self.border_mode = args.border_mode
		else:
			self.border_mode = 'valid'
		self.dropout_rate = args.dropout_rate
		if args.rnn_type == 'lstm':
			self.RNN = recurrent.LSTM
		elif args.rnn_type == 'gru':
			self.RNN = recurrent.GRU
		else:
			self.RNN = recurrent.SimpleRNN

		if args.is_training:
			self.embed_trainable = args.embed_trainable
			self.learning_rate = args.learning_rate
			self.clipnorm = args.clipnorm
			self.optimizer = args.optimizer
		else:
			self.embed_trainable=False

	def create_model_graph(self):
		# Take the inputs
		context = Input(shape=(None,), dtype='int32', name='story')
		question = Input(shape=(None,), dtype='int32', name='question')

		if self.qtype == 'wh2':
			qwh = Input(shape=(None,), dtype='float32', name='q_wh')
			qwh_next = Input(shape=(None,), dtype='float32',
					 name='q_wh_next')

		if self.char_embedding:
			context_ch = Input(shape=(None, self.maxwordlen,),
					   dtype='int32', name='story_char')
			question_ch = Input(shape=(None, self.maxwordlen,),
					    dtype='int32', name='question_char')


			# Character Embedding
			shared_char_embed = TimeDistributed(
				Embedding(output_dim=self.char_embed_dim,
					  input_dim=self.char_vocab_size))
			context_char_emb_init = (shared_char_embed)(context_ch)
			question_char_emb_init = (shared_char_embed)(question_ch)
			context_char_emb = Dropout(self.dropout_rate)(context_char_emb_init)
			question_char_emb = Dropout(self.dropout_rate)(question_char_emb_init)
			Maxpool = Lambda(lambda x: K.max(x, axis=2),
					 output_shape=lambda shape: (shape[0], shape[1], shape[-1]))
			shared_cnn_char = TimeDistributed(Convolution1D(self.char_embed_dim, 5,
									border_mode=self.border_mode,
									activation='tanh'))
			context_char_rep_init = (shared_cnn_char)(context_char_emb)
			context_char_rep_init = Maxpool(context_char_rep_init)
			question_char_rep_init = (shared_cnn_char)(question_char_emb)
			question_char_rep_init = Maxpool(question_char_rep_init)
			context_char_rep = Dropout(self.dropout_rate)(context_char_rep_init)
			question_char_rep = Dropout(self.dropout_rate)(question_char_rep_init)

		# word embedding
		shared_embed = Embedding(output_dim=self.embed_dim,
					 input_dim=self.vocab_size,
					 weights=[self.embed_mat],
					 trainable=self.embed_trainable,
					 mask_zero=True)
		context_emb_init = (shared_embed)(context)
		question_emb_init = (shared_embed)(question)
		context_word_emb = Dropout(self.dropout_rate)(context_emb_init)
		question_word_emb = Dropout(self.dropout_rate)(question_emb_init)

		# joint embedding
		if self.char_embedding:
			context_emb = merge([context_word_emb, context_char_rep],
					    mode='concat', concat_axis=-1)
			question_emb = merge([question_word_emb, question_char_rep],
					     mode='concat', concat_axis=-1)
		else:
			context_emb = context_word_emb
			question_emb = question_word_emb

		shared_rnn = Bidirectional(self.RNN(self.h, return_sequences=True),
					   merge_mode='concat')
        # [U x H]
		q_bi_rnn = (shared_rnn)(question_emb)
		q_bi_rnn = Dropout(self.dropout_rate)(q_bi_rnn)
		# [T x H]
        ctx_bi_rnn = (shared_rnn)(context_emb)
		ctx_bi_rnn = Dropout(self.dropout_rate)(ctx_bi_rnn)

		if self.qtype == 'wh2':
			q_1st_word = merge([qwh, q_bi_rnn],
					   mode='dot', dot_axes=(1, 1))
			q_2nd_word = merge([qwh_next, q_bi_rnn],
					   mode='dot', dot_axes=(1, 1))
		else:
			q_1st_word = Lambda(lambda x: x[:, 0, :],
					    output_shape=lambda shape: (shape[0], shape[2]))(q_bi_rnn)
			q_2nd_word = Lambda(lambda x: x[:, 1, :],
					    output_shape=lambda shape: (shape[0], shape[2]))(q_bi_rnn)
        # [T x 2H] * [U x 2H] => [T x U]
		q_attn = merge([ctx_bi_rnn, q_bi_rnn],
			       mode='dot', dot_axes=(2, 2))
		q_attn_sft = Activation('softmax')(q_attn)
		# [T x U] * [U x 2H] => [T x 2H]
        attended_q = merge([q_attn_sft, q_bi_rnn],
				   mode='dot', dot_axes=(2, 1))
        # [T x 2H] c [T x 2H] => [T x 4H]
		joint_enc = merge([ctx_bi_rnn, attended_q],
				      mode='concat', concat_axis=-1)
        # [T x 4H]
		joint_enc2 = TimeDistributed(Dense(4 * self.h,
							activation='sigmoid'))(joint_enc)
		# [T x 4H] * [T x 4H]
        joint_enc = merge([joint_enc2, joint_enc], mode='mul')
        # [T x 4H] => [T x 2H]
		encoding_pre = Bidirectional(self.RNN(self.h, return_sequences=True),
					     merge_mode='concat')(joint_enc)
		encoding_pre = Dropout(self.dropout_rate)(encoding_pre)
        # [T x 2H] ??? 
		enc_pre_dense = TimeDistributed(ThreeDOut(2 *
							  self.h))(encoding_pre)
        # ???
		self_attn_multi = merge([encoding_pre, enc_pre_dense],
					mode='dot', dot_axes=(2, 3))
		pooled_attn = Lambda(lambda x: K.max(x, axis=3),
				   output_shape=lambda shape: ((shape[0], shape[1], shape[2])))(
										self_attn_multi)
		pooled_attn_sft = Activation('softmax')(pooled_attn)
		# [T, T] * [T x 2H] => [T x 2H]
        self_matched_ctx = merge([pooled_attn_sft, encoding_pre],
					 mode='dot', dot_axes=(2, 1))
		# [T x 2H] c [T x 2H] => [T x 4H]
        joint_ctx_enc = merge([encoding_pre, self_matched_ctx],
					       mode='concat', concat_axis=-1)
        # [T x 2H] 
		joint_ctx_enc_gate = TimeDistributed(Dense(4 * self.h,
							activation='sigmoid'))(
							joint_ctx_enc)
        # [T x 2H]
		gated_joint_ctx_enc = merge([joint_ctx_enc_gate, joint_ctx_enc],
						     mode='mul')

        # [T x 2H]
		encoding = Bidirectional(self.RNN(self.h, return_sequences=True),
					 merge_mode='concat')(
					gated_joint_ctx_enc)
		encoding = Dropout(self.dropout_rate)(encoding)

		# Answer Pointers
		maxcol_q_attn = Lambda(lambda x: K.max(x, axis=1),
				       output_shape=lambda shape: (shape[0], shape[2]))(
										q_attn)
		norm_wts = Activation('softmax')(maxcol_q_attn)
        # [U x 1] * [U x 2H] => [1 x 2H]
		wtd_q1 = merge([norm_wts, q_bi_rnn],
			       mode='dot', dot_axes=(1, 1))
        # [1 x 6H]
		q_rep = merge([wtd_q1, q_1st_word, q_2nd_word],
			      mode='concat', concat_axis=-1)
		wtd_q = Dense(2 * self.h, activation='tanh')(q_rep)
		wtd_q = Dropout(self.dropout_rate)(wtd_q)

		ans_start_bef = merge([encoding, wtd_q],
				      mode='dot', dot_axes=(2, 1))
		ans_start_attn = Activation('softmax')(ans_start_bef)
		encoding_end = Bidirectional(self.RNN(self.h, return_sequences=True),
					     merge_mode='concat')(encoding)
		encoding_end = Dropout(self.dropout_rate)(encoding_end)
		ans_end_bef = merge([encoding_end, wtd_q],
				    mode='dot', dot_axes=(2, 1))
		ans_end_attn = Activation('softmax')(ans_end_bef)
		Expand = Lambda(lambda x: K.expand_dims(x, 1),
				output_shape=lambda shape: (shape[0], 1, shape[-1]))
		ans_pointer = merge([Expand(ans_start_attn),
				     Expand(ans_end_attn)],
				    mode='concat', concat_axis=1,
				    name='ans_start_end')

		if self.char_embedding:
			if self.qtype == 'wh2':
				model = Model(input=[context, question,
						     context_ch, question_ch,
						     qwh, qwh_next],
					      output=[ans_pointer])
			else:
				model = Model(input=[context, question,
						     context_ch, question_ch],
					      output=[ans_pointer])
		else:
			if self.qtype == 'wh2':
				model = Model(input=[context, question,
						     qwh, qwh_next],
					      output=[ans_pointer])
			else:
				model = Model(input=[context, question],
					      output=[ans_pointer])


		return model

	def compile_model(self, model):
		if self.optimizer == 'adam':
			opt = Adam(lr=self.learning_rate,
				   clipnorm=5.0)
		elif self.optimizer == 'adamax':
			opt = Adamax(lr=self.learning_rate,
				   clipnorm=5.0)
		elif self.optimizer == 'rmsprop':
			raise NotImplementedError
		else:
			raise NotImplementedError

		model.compile(optimizer=opt,
			      loss='categorical_crossentropy',
			      metrics=['accuracy'])
		return model

qa_model = QAModel()
model = qa_model.create_model_graph()
print(model.summary())