#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: Hou-Hou

'''
参考：https://www.jianshu.com/p/afd4312bbc68
'''
import sys
import os

import codecs
import pickle
import itertools
from collections import OrderedDict
import os
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from NER_IDCNN_CRF.model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils_NER import get_logger, make_path, clean, create_model, save_model
from utils_NER import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

flags = tf.app.flags   # 用tf.app.flags来定义参数，可以在flags里保存参数
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
# 因为Y是双标签，所以x也要用双标签来标注。BIOS是标注y的，不是x
# 文字有两重信息：1.文字本身的字向量：100维； 2.位置信息：20维
# ，急性呼吸道感染
# 0 1 2 2 2 2 2 3 逗号是0，开头是1，结尾是3，中间全是2
# 比如x急是0100四维，全连接20维，再加上原来的100维，100+20=120维。20就是做位置词的Embedding，用120维来代替一个x的输入
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")   # y有实体信息和位置信息，这里是标签的位置类型

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("batch_size",    20,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")  # 优化器，tf有9类优化器
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")  # 数据预处理embeding,char_dim是100，这里就是true
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")   # 碰到数字用0取代
flags.DEFINE_boolean("lower",       True,       "Wither lower case")   # 是否需要将字母小写，这个案例中，字符串不需要小写

flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")  # 最大epoch,建议5000-10000
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")     # 每100个batch输出损失
flags.DEFINE_string("ckpt_path",    "ckpt_biLSTM",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")  # 保存可视化摘要，保存流程图
flags.DEFINE_string("log_file",     "train.logs",    "File for logs")   # maps.pkl一般用来保存模型，这里保存字典
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")    # 保存字典的向量，训练集的正反向字典，将训练集隐射成word2vec的字典
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     os.path.join("data", "vec.txt"),  "Path for pre_trained embedding")
flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
#flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS    # 上面的参数保存在这里
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)     # vocab_size  词汇表大小
    config["char_dim"] = FLAGS.char_dim       # embeding_size 100  char_dim即是EMB_size
    config["num_tags"] = len(tag_to_id)       # BIO --> 数字
    config["seg_dim"] = FLAGS.seg_dim         # 增加的维度20，位置向量
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


# best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
#        evaluate(sess, model, "test", test_manager, id_to_tag, logger)
def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)   # [char, gold, pred]=[汉字，真实标签值，预测标签值]
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)     #写日志
    f1 = float(eval_lines[1].strip().split()[-1])

    # eval_lines[1]： 'accuracy:  96.29%; precision:  74.92%; recall:  63.73%; FB1:  68.87\n'

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()   # tf.assign(A, new_number): 这个函数的功能主要是把A的值变为new_number
            logger.info("new best dev f1 score:{:>.3f}".format(f1))   # 右对齐，保留3位小数
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets：将原始数据加载为：[[['海', 'O'], ['钓', 'O'],...,], [第二个句子],..., [] ]格式
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)  # 20864
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)      # 2318
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)    # 4636

    # Use selected tagging scheme (IOB / IOBES)：将句子处理成：IOB/IOBES格式
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)

    # create maps if not exist
    # FLAGS.map_file：文字、标签----数字
    if not os.path.isfile(FLAGS.map_file):  # map_file不存在时：写进去
        # create dictionary for word
        if FLAGS.pre_emb:   # Wither use pre-trained embedding：因为预训练模型是根据维基百科训练的，汉字可能与训练集不同，因此将预训练集中训练集不存在的汉字频率设置为0
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]   #  return dico, char_to_id, id_to_char  dico = {单词，词频}
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])   # 提取test_sentences中的单词，转换为一维list
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)

        # 保存
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:  # map_file存在时：读出来
        # 词到ID,标记到ID。pickle用来打开maps.pkl文件
        # 这四个值就是正反向字典，长度分别是
        #  2678        2678         51         51
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # id_to_tag：{0: 'O', 1: 'I-ORG', 2: 'B-LOC', 3: 'E-LOC', 4: 'B-ORG', 5: 'E-ORG', 6: 'I-LOC', 7: 'I-PER', 8: 'B-PER', 9: 'E-PER', 10: 'S-LOC', 11: 'S-PER', 12: 'S-ORG'}

    # prepare data, get a collection of list containing index
    # [string，chars，segs，tags]
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (len(train_data), len(dev_data), len(test_data)))

    # 创建batch data，根据batch中句子的max_length，处理成相同长度的句子，用0补齐
    train_manager = BatchManager(train_data, FLAGS.batch_size)   # num_batch：20864/20=1043.2=1044
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)

    # make path for store logs and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("logs", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)   # 在控制台输出日志

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片

    steps_per_epoch = train_manager.len_data   # 1044

    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)   # 通过create_model加载 Model里面的模型
        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)   # batch：输入值格式？  step通过global_step在整个模型训练过程中传递
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def evaluate_line():
    config = load_config(FLAGS.config_file)
    logger = get_logger(FLAGS.log_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)   # python的pickle模块实现了基本的数据序列和反序列化,load从字节对象中读取被封装的对象，并返回
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger, False)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

                line = input("请输入测试句子:")
                result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)


def main(_):

    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        train()
    else:
        evaluate_line()
    # train()

if __name__ == "__main__":
    tf.app.run(main)



