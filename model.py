# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, positionwise_feedforward_and_residual_and_norm, \
    positional_encoding, multihead_attention_and_residual_and_norm, \
    label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        # embeddings:[vocab_size, d_model]
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    """
    对于encoder而言,有1处attention:
    0. x = position embedding + word embedding
    1. self-attention
    2. residual+layer normalization
    3. feed forward
    4. residual+layer normalization
    """
    def encode(self, xs, training=True):
        '''
        xs:(x, seqlens, sents1)
            x:[N,T1]
            seqlens:[N]
            sents1:[N]
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # embedding
            # x:[N,T1], embeddings:[vocab, d_model]
            # enc:[N, T1, d_model]
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale , *sqrt(512)

            # enc:[N, T1, d_model]
            # position_encoding:[N, T1, d_model]
            position_encoding = positional_encoding(enc, self.hp.maxlen1)

            """
            注意:position encoding直接加在原始encoding中
            """
            # enc:[N, T1, d_model]
            enc += position_encoding
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training) # dropout_rate默认为0.3

            """
            encoder中每个block由2部分组成:
            1. self-attention
            2. feed-forward
            """
            ## Blocks
            # encoder是编码器的堆叠(论文中堆叠了6层):
            for i in range(self.hp.num_blocks):
                # 注意:variable_scope不同,encoder各变量并不共享参数!!!
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # 1.self-attention
                    # enc:[N, T1, d_model]
                    enc = multihead_attention_and_residual_and_norm(queries=enc,
                                                                    keys=enc,
                                                                    values=enc,
                                                                    num_heads=self.hp.num_heads,  # 多头attention
                                                                    dropout_rate=self.hp.dropout_rate,
                                                                    training=training,
                                                                    causality=False)
                    # 2.positionwise feed forward
                    # enc:[N, T1, d_model]
                    enc = positionwise_feedforward_and_residual_and_norm(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        # 最后一层的输出当成memory
        # enc:[N, T1, d_model]
        memory = enc
        return memory, sents1 #sents1:好像没啥用?

    """
    decoder中,每个block由3部分组成:
    1. self-attention
    2. encoder-decoder attention
    3. feed forward
    
    对于decoder而言,有2处attention:
    1. self-attention
    2. residual+layer normalization
    3. encoder-decoder attention
    4. residual+layer normalization
    5. feed forward
    6. residual+layer normalization
    """
    def decode(self, ys, memory, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)

        ys: tuple of
            decoder_input: int32 tensor. (N, T2)
            y: int32 tensor. (N, T2)
            y_seqlen: int32 tensor. (N, )
            sents2: str tensor. (N,)

        Returns
        decoder_input: int32 tensor. (N, T2)
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32
        y: (N, T2). int32
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # embedding
            # embeddings:[vocab_size, d_model]
            # decoder_input:[N,T2]
            # dec:[N,T2,d_model]
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model), 与encode用的是同一embedding矩阵
            dec *= self.hp.d_model ** 0.5  # scale, *sqrt(d_model)

            """
            注意:position encoding直接加在原始encoding中
            """
            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                # 注意:variable_scope不同,decoder各变量并不共享参数!!!
                # 并且decoder不与encoder共享参数
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time), 先是decoder自身的self-attention
                    # dec:[N,T2,d_model]
                    dec = multihead_attention_and_residual_and_norm(queries=dec,
                                                                    keys=dec,
                                                                    values=dec,
                                                                    num_heads=self.hp.num_heads,
                                                                    dropout_rate=self.hp.dropout_rate,
                                                                    training=training,
                                                                    causality=True, # 需要用time_step来mask那些将来的token embedding
                                                                    scope="self_attention")

                    # Vanilla encoder-decoder attention, 再用decoder-encoder attention
                    # dec:[N,T2,d_model]
                    # memory:[N, T1, d_model], memory可以看成encoder的所有timestep生成的hidden_state
                    dec = multihead_attention_and_residual_and_norm(queries=dec,
                                                                    keys=memory,
                                                                    values=memory,
                                                                    num_heads=self.hp.num_heads,
                                                                    dropout_rate=self.hp.dropout_rate,
                                                                    training=training,
                                                                    causality=False,
                                                                    scope="vanilla_attention")
                    ### Feed Forward && add and norm
                    # dec:[N,T2,d_model]
                    dec = positionwise_feedforward_and_residual_and_norm(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        # embeddings:[vocab_size, d_model]
        # wegihts: [d_model, vocab_size]
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size), 此处与encoder时的lookuptable共享embedding

        # dec:[N,T2,d_model]
        # wegihts: [d_model, vocab_size]
        # logits:[N, T2, vocab_size]
        # 注意:为何不用Linear project层,因为无法共享embedding
        """
        einsum:Einstein summation 
        ntd,dk -> ntk 
        即类比于矩阵相乘: [N, T2, d_model] * [d_model, vocab_size] => [N, T2, vocab_size]
        等价于:
        logits = tf.matmul(dec, weights) # 注意:tensorflow 不支持这种维度不一致的matmul
        
        换成matmul写法(比较啰嗦):
        input_shape = tf.shape(dec)
        input2 = tf.reshape(dec,[-1, input_shape[-1]])
        # [N*T, vocab_size]
        logit_mat2 = tf.matmul(input2, weights)
        logit_mat = tf.reshape(logit_mat2, [input_shape[0], input_shape[1], tf.shape(weights)[-1]])
        """
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size) , 这个就是论文中的线性映射层
        # logits:[N, T2, vocab_size]
        # y_hat:[N, T2]
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        # y: (N, T2). int32
        # sents2: (N,). string.
        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''

        # forward
        # xs:
        # x:[N, T1]
        # seqlens:[N]
        # memory:[N, T1, d_model]
        # sents1:[N]
        memory, sents1 = self.encode(xs)

        # ys:
        # decoder_input: [N, T2]
        # y:[N, T2]
        # seqlens:[N]
        # memory:[N, T1, d_model]
        # sents2:[N]

        """
        注意:不管decoder里有多少层block,用的memory都是相同的,即encoder产生的
        """
        # logits:[N, T2, vocab_size], float32
        # preds:[N, T2], int32
        logits, preds, y, sents2 = self.decode(ys, memory)
        logging.info("logits:{} preds:{}".format(logits, preds))

        # train scheme
        # y:[N, T2]
        # y_:[N, T2, vocab_size]
        # 标签的laplace平滑
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))

        """
        感觉这里y_与logits有相同的shape
        """
        # logits:[N, T2, vocab_size]
        # y_:[N, T2, vocab_size]
        # cross_entropy:[N, T2]
        # cross_entropy:[N, T2]
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_) # 梯度可以回传到logist, label, 从而回传到embedding
        logging.info("logits:{} labels:{} cross_entropy:{}".format(logits, y_, cross_entropy)) # logits:(?, ?, 32000), labels:(?,?,32000)
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_) # 梯度只能回传到logit
        # y:[N, T2]
        # nonpadding:[N, T2]
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>, 注意:<pad>的id为0
        # cross_entropy:[N, T2]
        # nonpadding:[N, T2]
        # loss:scalar
        loss = tf.reduce_sum(cross_entropy * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7) # 针对batch*timestep的平均,而不是batch的平均

        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        # xs[0]: [N, T1]
        # decoder_inputs: [N, 1], 值都是<s>
        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, training=False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):# 注意:这里并没有beamsearch
            # logits:[N, T2, vocab_size]
            # y_hat:[N, T2].int32
            # y: (N, T2).int32
            logits, y_hat, y, sents2 = self.decode(ys, memory, training=False) # y_hat的长度不断增加
            # y_hat:[N, T2]
            # cur_token:[N,]
            """
            不明白为何将token id相加?答案:其实这里只是降维而己,每次只是预测出一个token_id, 所以相加之后还是自己
            """
            cur_token = tf.reduce_sum(y_hat, axis=1) # cur_token:[N,],
            if _ ==0:logging.info("y_hat:{} cur_token:{}".format(y_hat, cur_token))

            """
            如果当前预测的字符为padding,则终止预测
            即当此batch中所有的样本都预测完毕时,才终止,否则将一直预测
            """
            if cur_token == self.token2idx["<pad>"]: break
            # 生成新的decoder_inputs
            _decoder_inputs = tf.concat((decoder_inputs, y_hat), axis=1) # 将预测出的"I"追加, 即:"SOS I"
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample, 随机选择一个句子作为监控
        # y_hat:[N, T2].int32
        # y_hat[0]:N
        sent_index = tf.random_uniform((), minval=0, maxval=tf.shape(y_hat)[0] - 1, dtype=tf.int32)
        sent1 = sents1[sent_index] # 原文
        pred = convert_idx_to_token_tensor(y_hat[sent_index], self.idx2token)
        sent2 = sents2[sent_index] # 译文

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

