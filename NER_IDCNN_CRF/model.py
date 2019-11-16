# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood   # 条件随机场
from tensorflow.contrib.crf import viterbi_decode       # 隐马尔科夫链
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn
from utils_NER import result_to_json
from data_utils import create_input, iobes_iob

# 双向lstm或IdCNN模型，找到x,y. y是双标签（B-LOC，B、LOC两个标签），x是文字word2vec映射成的词向量。
# 如何拟合x.y：拟合之前第一步提取x的特征，用BiLstm或idCNN对x做特征提取，+ 分类器（crf条件随机场）
# BiLstm or idCNN + crf
# idCNN与cnn的区别是，idCNN的卷积核是扁的：找一句话之间的关系可以用扁的，
# 好处：可以有效地抗噪音：完形填空时，扁的卷积核它只会扫当前这句话，不会把上下文卷进来，抗的是上下文的躁
# CNN和RNN本质上没有太大差别，都是把局部的相关性体现出来，CNN体现在空间上，RNN体现在时间时序上

# crf：条件随机场。跟rnn很类似，提供了一个分类结果，当然它也可以做特征提取。它的分类需要算一个联合概率
# 第一步，找到x,y
# 第二步，对x做特征提取、特征工程（之前所有的resnet等都是为特征工程服务的），对y做one_hot向量（或二分类）
# 第三步，去拟合，分类

# crf_log_likelihood (#likelihood似然，一般加似然的就是损失函数

class Model(object):
    def __init__(self, config, is_train=True):

        self.config = config
        self.is_train = is_train
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]   # embeding_size 100  char_dim即是EMB_size
        self.lstm_dim = config["lstm_dim"]   # 100
        self.seg_dim = config["seg_dim"]     # 增加的维度20，位置向量

        self.num_tags = config["num_tags"]     # 13，tag的标签个数：B、I、O等的标签数  len(tag_to_id)
        self.num_chars = config["num_chars"]   # 4412，词汇表大小，即vocab_size     config["num_chars"] = len(char_to_id)
        self.num_segs = 4                      # 0，1，2，3：0是不需要的字，1是第一个，2是中间的，3是最后一个

        self.global_step = tf.Variable(0, trainable=False)   # train时使用
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()   # xavier_initializer迭代器，效率高，和global_initializer类似

        # 1. add placeholders for the model
        # batch_size是20
        self.char_inputs = tf.placeholder(dtype=tf.int32,shape=[None, None], name="ChatInputs") # 这个是20*100？    batch_size是20
        # seg_input只有四个值，0、1、2、3
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")  # 这个是20*20，0-3映射成20    batch_size是20

        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")  # 这个是20*1,y值  ？？？
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))             # 返回：-1，0，1
        length = tf.reduce_sum(used, reduction_indices=1)    # reduction_indices=1，是第1维对应位置相加。
        # 二维的东西，降掉一维，算整个长度是多少
        self.lengths = tf.cast(length, tf.int32)             # 非train时使用，同self.global_step：输入序列的实际长度（可选，默认为输入序列的最大长度） 120
        self.batch_size = tf.shape(self.char_inputs)[0]      # 20*100，第0个就是20
        self.num_steps = tf.shape(self.char_inputs)[-1]
        
        
        # Add model type by crownpku， bilstm or idcnn
        self.model_type = config['model_type']
        # parameters for idcnn
        # idcnn后面连的是膨胀卷积，好处：有些图像比较小的时候，不希望挤到一起。防止欠拟合
        # 一种方法是，把图像做膨胀，另一种方法是将卷积核做膨胀。一般是feature_map做膨胀，卷积核不膨胀
        # 由3*3变成5*5，中间补0
        self.layers = [
            {
                'dilation': 1   # 膨胀卷积：膨胀卷积核尺寸 = 膨胀系数*（原始卷积核尺寸-1）+1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        self.filter_width = 3     # 卷积核宽3,卷积核的高没有写，所以高是1，1*3，卷积核是扁的
        self.num_filter = self.lstm_dim    # 100，卷积核个数即为lstm连接隐层的个数，就是卷积的通道数输出的
        self.embedding_dim = self.char_dim + self.seg_dim   # 字向量的维度+词长度特征维度   embedding_size 120=100+20
        self.repeat_times = 4    # 重复的次数是4，4层卷积网络 深度3*4=12层，重复的是self.layers
        self.cnn_output_width = 0   # 输出的宽度实际上是2000多维，这里初始化为0

        # 2.输入：embeddings for chinese character and segmentation representation
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)  # (?,?,120)

        # 3.构建网络
        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)   # (?,?,120)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)
        
        elif self.model_type == 'idcnn':
            # apply dropout before feed to idcnn layer
            # 120个里面随机删掉一部分，内存不删，删里面的值
            # dropout在输入层、输出层、隐层都可以做
            model_inputs = tf.nn.dropout(embedding, self.dropout)   # (?,?,120)

            # ldcnn layer
            # ldcnn layer：特征提取 膨胀卷积
            # model_inputs是120维的，如果做了dropout，就剩60维了
            model_outputs = self.IDCNN_layer(model_inputs)
            # 输入（100+20）个——卷积--》（100个通道---》3次膨胀（100）以上循环4次）

            # logits for tags
            self.logits = self.project_layer_idcnn(model_outputs)
        
        else:
            raise KeyError

        # 4. 计算loss：CRF层
        self.loss = self.loss_layer(self.logits, self.lengths)

        # 5.定义优化器
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError

            # apply grad clip to avoid gradient explosion
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size], 
        """

        embedding = []
        with tf.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            # self.num_chars=3538， self.char_dim=100维 ，char_lookup字符查找
            self.char_lookup = tf.get_variable(
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],   # 4412*100  num_chars：词汇表大小，即vocab_size；char_dim即是EMB_size
                    initializer=self.initializer)
            # input_data的维度是：batch_size * num_steps
            # 输出的input_embedding的维度是：batch_size * num_steps * EMB_size
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))   # 把input映射成embedding

            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],   # self.num_segs=4, self.seg_dim=20 ，4*20的
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
                    # seg_input只有四个值，0、1、2、3
            embed = tf.concat(embedding, axis=-1)   # 组成120维向量
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(   # LSTMCell基于LSTM：搜索空间奥德赛的输入和忘记门的扩展。
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)     # 输入序列的实际长度（可选，默认为输入序列的最大长度）
        return tf.concat(outputs, axis=2)  # 将两个LSTM的输出拼接为一个张量：在第2维上进行concat，即拼接各个特征
    
    # IDCNN layer
    def IDCNN_layer(self, model_inputs, name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)   # 将三维变为CNN的输入格式：四维  [长，宽，深，特征/filters]
        # tf.expand_dims会向tensor中插入一个维度，插入位置就是参数代表的位置(维度从0开始)
        # shape由[?,?,120]变成[?,1,?,120]，最后一维是特征，即embedding
        reuse = False
        if not self.is_train:  # 不是train时，reuse = True
            reuse = True
        with tf.variable_scope("idcnn" if not name else name):
            shape = [1, self.filter_width, self.embedding_dim, self.num_filter]  # [1, 3, 120, 100]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim, self.num_filter],   # embedding_dim：字向量的维度+词长度特征维度
                initializer=self.initializer)
            
            """                  
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
                               height是默认1，width是句子长度，通道是100维
            """
            # model_inputs为四维向量：后面三个维度对应一个节点矩阵，第一维对应一个输入 batch
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",         # SAME：全0填充；VALID：不填充
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            # 多次卷积，就会将膨胀的时候单次没有卷到的数据在下次卷到
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i, reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter, self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])

                        # 膨胀卷积：插入rate-1个0 这里三层{1,1,2}相当于前两个没有膨胀
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv

            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))    # 相当于tf.matmul(x, weights) + biases

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])
    
    # Project layer for idcnn by crownpku
    # Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))
                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        # num_steps是句子长度；project_logits 是特征提取并全连接后的输出
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            # start_logits=[batch_size,1,num_tags+1]  (?,1,14)
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)

            # pad_logits=[batch_size,num_steps,1]  (?,?,1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)  # (?,?,14)

            # logits=[batch_size,num_steps+1,num_tags+1]
            logits = tf.concat([start_logits, logits], axis=1)   # (?,?,14)

            # targets=[batch_size,1+实际标签数]
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(               # 条件随机场的输出  状态转移阵
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)

            # logits是模型的特征输出；targets是label; trans是条件随机场的输出

            # crf_log_likelihood 在一个条件随机场里计算标签序列的log-likelihood
            # inputs:一个形状为 [batch_size,max_seq_len,num_tags]的tensor， 一般使用BiLSTM处理之后输出转换为他要求的形状作为CRF层的输入
            # tag_indices：A [batch_size, max_seq_len] matrix of tag indices for which we compute the log-likelihood.
            # sequence_lengths:一个形状为[batch_size]的向量，表示每个序列的长度
            # transition_params:形状为[num_tags,num_tags]的转移矩阵
            # log_likelihood:标量，log-likelihood
            # 注意：由于条件随机场有标记，故真实维度+1
            # inputs=[char_inputs,seg_inputs]
            # 高：3 血：22 糖：23 和：24 高：3 血：22 压：25                     char_inputs=[3,22,23,24,3,22,25]
            # 高血糖 和 高血压 seg_inputs 高血糖=[1,2,3] 和=[0] 高血压=[1,2,3]   seg_inputs=[1,2,3,0,1,2,3]

            # tf.contrib.crf.crf_log_likelihood()  likelihood似然，一般加似然的就是损失函数
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)  # lengths是字的个数，logits是模型特征
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]                            # （7，13）
            pad = small * np.ones([length, 1])                # （7，1）
            logits = np.concatenate([score, pad], axis=1)     # 13 + 1 维  （7，14）
            logits = np.concatenate([start, logits], axis=0)  # （8，14）
            # 由显式序列logits和状态转移阵matrix，求隐藏序列的最大概率路径，也即最短路径
            viterbi, _ = viterbi_decode(logits, matrix)

            paths.append(viterbi[1:])   # 去除第一个值start
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)    # 维特比算法解码
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]    # 去除后面padding的0，得到句子实际长度
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])          # 真实的标签值
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])   # 预测的标签值
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))    # [汉字，真实标签值，预测标签值]
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
