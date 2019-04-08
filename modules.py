# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Building blocks for Transformer
'''

import numpy as np
import tensorflow as tf

def layer_normalization(inputs, epsilon = 1e-8, scope="layer_normalization"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # inputs: [N, T_q, d_model].
        inputs_shape = inputs.get_shape() # (N, T_q, d_model)
        params_shape = inputs_shape[-1:] # 取出最后的维度,(d_model,)

        # inputs: [N, T_q, d_model].
        # mean: [N, T_q, 1],只在最后一个维度上进行求平均
        # variance: [N, T_q, 1],只在最后一个维度上进行求方差
        mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
        # beta:[d_model,]
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        # gamma:[d_model,]
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        # inputs: [N, T_q, d_model].
        # mean: [N, T_q, 1]
        # normalized: [N, T_q, d_model].
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) ) # (x-mu)/sigma
        """
        注意:此处的gamma在各个维度上的值并不相同,即各维度上不共享
        """
        # gamma:[d_model,]
        # normalized: [N, T_q, d_model].
        # beta:[d_model,]
        outputs = gamma * normalized + beta
        
    return outputs

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_units),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings

def scaled_dot_product_attention(Q, K, V,
                                 causality=False,
                                 dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.

    :return query_key_interaction:[N, T_q, d_v]
    '''
    """
    计算每个q对所有k_i的score
    如q=[i,love,nlp]
    
    score_i = q*k(i)/sqrt(d_k)
    score_i = softmax(score_i)
    value = sum_{i}{score_i* value_i}
    """

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        # Q: [N, T_q, d_k]
        # K: [N, T_k, d_k] => [N, d_k, T_k]
        # query_key_interaction:[N, T_q, T_k]
        query_key_interaction = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        # query_key_interaction:[N, T_q, T_k]
        query_key_interaction /= d_k ** 0.5

        # key masking
        # Q:[N, T_q, d_k]
        # K:[N, T_k, d_k]
        # query_key_interaction:[N, T_q, T_k]
        # 对于key mask的话, 将key padding成outputs的维度
        query_key_interaction = mask(query_key_interaction, Q, K, type="key")

        # causality or future blinding masking
        """
        生成query_key_interaction关于时间的掩码下三角矩阵
        query_key_interaction:[N, T_q, T_k]
        例:  
        [[-1.4095545   0.          0.        ]
         [ 0.526246   -0.11131065  0.        ]
         [ 0.80647576 -0.886015   -0.04653838]
         [ 1.073006   -0.6044851  -0.7388869 ]]
         
        即时间T_q=2时,只能与时间为T_k=1或T_k=2的交互
          时间T_q=3时,只能与时间为T_k=1或T_k=2, T_k=3的交互
        
        由于在代码中,padding的是一个非常大的负数,因此经过softmax之后,会变成0
        """
        if causality: # 因果关系
            query_key_interaction = mask(query_key_interaction, type="future")

        # softmax
        # query_key_interaction_norm:[N, T_q, T_k]
        query_key_interaction_norm = tf.nn.softmax(query_key_interaction)
        # attention:[N, T_k, T_q]
        attention = tf.transpose(query_key_interaction_norm, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1)) # 取第一个元素,进行展示

        # query masking
        # Q:[N, T_q, d_k]
        # K:[N, T_k, d_k]
        # query_key_interaction_norm:[N, T_q, T_k]
        # 对于query mask的话, 将query padding成outputs的维度
        query_key_interaction_norm = mask(query_key_interaction_norm, Q, K, type="query")

        # dropout
        # query_key_interaction_norm:[N, T_q, T_k]
        query_key_interaction_norm = tf.layers.dropout(query_key_interaction_norm, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        # query_key_interaction_norm:[N, T_q, T_k]
        # V:[N, T_k, d_v].
        # attentioned_value:[N, T_q, d_v]
        attentioned_value = tf.matmul(query_key_interaction_norm, V)  # (N, T_q, d_v)

    return attentioned_value

def mask(inputs, queries=None, keys=None, type=None):
    """Masks paddings on keys or queries to inputs
    inputs: 3d tensor. (N, T_q, T_k)
    queries: 3d tensor. (N, T_q, d)
    keys: 3d tensor. (N, T_k, d)

    e.g.,
    >> queries = tf.constant([[[1.],
                        [2.],
                        [0.]]], tf.float32) # (1, 3, 1)
    >> keys = tf.constant([[[4.],
                            [0.]]], tf.float32)  # (1, 2, 1)
    >> inputs = tf.constant([[[4., 0.],
                              [8., 0.],
                              [0., 0.]]], tf.float32) #(1,3,2)
    >> mask(inputs, queries, keys, "key") # (1, 3, 2)
    array([[[ 4.0000000e+00, -4.2949673e+09],
        [ 8.0000000e+00, -4.2949673e+09],
        [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)

    >> inputs = tf.constant([[[1., 0.],
                             [1., 0.],
                             [1., 0.]]], tf.float32) # (1,3,2)
    >> mask(inputs, queries, keys, "query") # (1,3,2)
    array([[[1., 0.],
        [1., 0.],
        [0., 0.]]], dtype=float32)
    """
    padding_num = -2 ** 32 + 1 # -4294967297
    if type in ("k", "key", "keys"):
        # 对于key的话, padding成input的维度
        # Generate masks
        # keys: [N, T_k, d]
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        masks = tf.expand_dims(masks, 1) # (N, 1, T_k)
        # masks:[N, 1, T_k]
        # queries: [N, T_q, d]
        masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

        # Apply masks to inputs
        # inputs: [N, T_q, T_k]
        paddings = tf.ones_like(inputs) * padding_num # 注意, padding的不是0,而是一个比较小的数
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
    elif type in ("q", "query", "queries"):
        # 对于query的话, padding成input的维度
        # Generate masks
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
        masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

        # Apply masks to inputs
        outputs = inputs*masks
    elif type in ("f", "future", "right"):
        """
        生成inputs的关于时间的掩码下三角矩阵
        
        此处是因果推断,即预测T时刻时不能提前看到T时刻的标签
        input: [[-1.4095545  -0.5366828  -0.5652379 ]
                [ 0.526246   -0.11131065  0.26350743]]
        tril: [[-1.4095545   0.          0.        ]
               [ 0.526246   -0.11131065  0.        ]] 
        """
        # inputs: [N, T_q, T_k]
        diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
        # 生成diag_vals的下三角矩阵
        # tril:[T_q, T_k]
        tril = tf.linalg.LinearOperatorLowerTriangular(tril=diag_vals).to_dense()  # (T_q, T_k)
        # masks:[N, T_q, T_k]
        masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

        paddings = tf.ones_like(masks) * padding_num
        # outputs: [N, T_q, T_k]
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")


    return outputs

def multihead_attention_and_add_and_norm(queries, keys, values,
                                         num_heads=8,
                                         dropout_rate=0,
                                         training=True,
                                         causality=False,
                                         scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)

    =====================
    注意:T_q与T_k的值不一定会相同,所以会有padding
    procedures:
    1. self-attention
    2. add residual
    3. layer normaliztion
    =====================
    '''
    # queries: [N, T_q, d_model].
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        """
        Query vector, Key vector, Value vector三者向量维度为64(paper中的参数),小于原始embedding_size(512).
        注意:感觉此处与原始paper中的略有不同,原始paper中,输入向量的维度(512)>映射后的向量维度(64),而此处是相等的
        
        q=X*Wq 
        k=X*Wk
        v=X*Wv
        """
        # Linear projections, 将query, key, value映射成指定维度的向量
        # queries:[N, T_q, d_model], units=d_model
        # Q:[N, T_q, d_model]
        Q = tf.layers.dense(inputs=queries, units=d_model, use_bias=False) # (N, T_q, d_model), 里面是matrix_W_Q
        K = tf.layers.dense(inputs=keys, units=d_model, use_bias=False) # (N, T_k, d_model), matrix_W_k
        V = tf.layers.dense(inputs=values, units=d_model, use_bias=False) # (N, T_k, d_model), matrix_W_v
        
        # Split and concat
        # 将原始的Q在第2维分割成num_heads=h份,然后在第0维拼接
        Q_ = tf.concat(tf.split(value=Q, num_or_size_splits=num_heads, axis=2), axis=0) # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(value=K, num_or_size_splits=num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(value=V, num_or_size_splits=num_heads, axis=2), axis=0) # (h*N, T_k, d_model/h)

        """
        score_i = q*k(i)/sqrt(d_k)
        score_i = softmax(score_i)
        value = sum_{i}{score_i* value_i}
        
        此处的mulit-head attention设计的确巧妙!
        """
        # Attention
        # Q_:(h*N, T_q, d_model/h)
        # K_:(h*N, T_q, d_model/h)
        # V_:(h*N, T_q, d_model/h)
        # attentioned_value:[N=h*N, T_q, T_v=d_model/h]
        attentioned_value = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        """
        注意:此处与原paper有所不同,原paper中,在concat向量之后,还会将concated_vector乘以Wo,但此处没有乘以Wo
        """
        # 直接将multi_head的输出concat拼接起来
        # Restore shape
        # attentioned_value:[h*N, T_q, d_model/h] -> (N, T_q, d_model)
        attentioned_value = tf.concat(tf.split(attentioned_value, num_heads, axis=0), axis=2) # (N, T_q, d_model)
              
        # Residual connection,残差连接
        # queries: [N, T_q, d_model].
        # attentioned_value: [N, T_q, d_model].
        attentioned_value += queries
              
        # layer normalize
        # attentioned_value: [N, T_q, d_model].
        attentioned_value = layer_normalization(attentioned_value)
 
    return attentioned_value

def positionwise_feedforward_and_add_and_norm(inputs, num_units, scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs

    Observe that during this step, vector representations of tokens don’t “interact” with each other.
    It is equivalent to run the calculations row-wise and stack the resulting rows in a matrix.
    注意:这个是每个位置上的词,做自己的feed-forward,即各token间并没有发生交互.
    procedure:
    1. feed-forward (两层全连接,中间加relu激活函数)
    2. residual (add)
    3. layer normalization (norm)
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        # inputs: [N, T, C]. N=batch_size, T=time_step, C=channel
        # outputs: [N, T, num_units[0]].
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu)

        # Outer layer
        # outputs: [N, T, num_units[1]].
        outputs = tf.layers.dense(outputs, units=num_units[1])

        # Residual connection
        # outputs: [N, T, num_units[1]].
        outputs += inputs
        
        # Normalize
        outputs = layer_normalization(outputs)
    
    return outputs

def label_smoothing(inputs, epsilon=0.1):
    """
    有点像laplace平滑,将给为0的标签一个比较小的概率
    :param inputs:
    :param epsilon:
    :return:
    """
    '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
    inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
    epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)
    
def positional_encoding(inputs,
                        maxlen,
                        masking=True,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.

    说明:
    偶数位置:
        PE(pos,2i)  =sin(pos/power(10000,2i/d_model))
    奇数位置:
        PE(pos,2i+1)=cos(pos/power(10000,2i/d_model))
    '''

    E = inputs.get_shape().as_list()[-1] # static, tuple
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), multiples=[N, 1]) # (N, T)

        # First part of the Poistion Embedding function: sin and cos argument
        # 偶数就不减,奇数减1
        # [maxlen, E]
        position_enc = np.array([
            [pos / np.power(10000, (i-i%2)/E) for i in range(E)] for pos in range(maxlen)
        ])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # (maxlen, E)

        # lookup
        # position_enc:[maxlen, E]
        # position_ind:[N, T]
        # outputs:[N, T, E]
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            # inputs:[N,T,E]
            # outputs:[N,T,E]
            # inputs中mask=0的地方,还是填0
            outputs = tf.where(tf.equal(inputs, 0), x=inputs, y=outputs)

        return tf.to_float(outputs)

def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    """
    通过代码模拟,可以看出其lr先急速增加至init_lr, 后缓慢减小
    init_lr * warmup^0.5 * min(step*warmup^(-1.5), step^(-0.5))
    """
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)