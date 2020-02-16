#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import timeit
from preprocess import *


# In[ ]:





# In[ ]:


def get_train_test(imgsamples, train_ratio = 0.9):

    idx = np.arange(len(imgsamples)) 
    np.random.shuffle(idx) 
    idx_1 = idx[:int(train_ratio*len(imgsamples))] 
    idx_2 = idx[int(train_ratio*len(imgsamples)):]
    img_train = [imgsamples[k] for k in idx_1]
    img_test = [imgsamples[k] for k in idx_2]
    
    return img_train, img_test


# In[ ]:


def prepare_batch(data, path, num = 1, is_train = True):
    
    allnames = list(data.keys())
    if is_train:
        idx = np.random.choice(len(allnames), num, replace=False)
    else:
        idx = np.arange(len(allnames))
    imgbatch = [allnames[k] for k in idx]
    labelbatch = [data[k] for k in imgbatch]
    
    X_batch = []
    y_batch = []
    
    for i in range(len(imgbatch)):
        img = plt.imread(path + imgbatch[i])
        img_bag = cut_img(resize_img(img, output_h = 800, output_w = 600), h_sep = 10)
        
        label_bag = []
        h_r, w_r = img.shape[0]/800, img.shape[1]/600
        boxes = [[k[1]//h_r, k[3]//h_r] for k in labelbatch[i]]
        for j in range(80):
            top = j*10
            bottom = (j+1)*10
            cur_label = 0
            for box in boxes:
                if bottom >= box[0] and top <= box[1]:
                    cur_label = 1
            label_bag.append(cur_label)
            
        X_batch.append(img_bag)
        y_batch.append(label_bag)
        
    X_batch = np.stack(X_batch, axis = 0)
    y_batch = np.stack(y_batch, axis = 0)
    y_batch = y_batch[:, :, np.newaxis]
    return X_batch, y_batch, imgbatch


# In[ ]:





# In[ ]:


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# In[ ]:


# here X is in shape of [batch, 80, H=10, W=600, C=1]

def build_cnn(X, H=10, W=600, C=1):
    
    X_reshape = tf.reshape(X, [-1, H, W, C]) # [batch*80, H=10, W=600, C=1]
    
    kernels = [5, 5, 3, 3, 3]
    filters = [1, 32, 64, 128, 128, 256]
    strides = [(2,2), (2,2), (2,2), (1,2), (1,2)]
    layers = 5
    
    pool = X_reshape
    for i in range(layers):
        kernel = tf.Variable(tf.truncated_normal([kernels[i], kernels[i], filters[i], filters[i+1]], stddev=0.1))
        conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides= [ 1, 1, 1, 1])
        conv_norm = tf.layers.batch_normalization(conv, training=training)
        relu = tf.nn.relu(conv_norm)
        pool = tf.nn.max_pool(relu, (1, strides[i][0], strides[i][1], 1), (1, strides[i][0], strides[i][1], 1), 'VALID')
        
    return pool

# cnn_output is in shape of [B*80, 1, 18, 256]


# In[ ]:


# cnn_output is in shape of [B*80, 1, 18, 256]

def build_rnn_hor(cnn_output):
    
    rnn_input = tf.squeeze(cnn_output, axis=[1]) # rnn_intput is in shape of [B*80, 18, 256]
        
    n_unit = 128
    cells = [tf.contrib.rnn.LSTMCell(num_units=n_unit) for _ in range(3)] # 2 layers

    # stack basic cells
    stacked = tf.contrib.rnn.MultiRNNCell(cells)

    # bidirectional RNN
    # BxTxF -> BxTx2H
    ((fw, bw), (fw_s, bw_s)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, 
                                                               inputs=rnn_input, dtype=rnn_input.dtype, 
                                                               scope = 'rnn_hor')

    #####output_concat = tf.concat([fw_s[-1][1], bw_s[-1][1]], 1)
    
    '''
    # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
    output_concat = tf.concat([fw, bw], 2)
    output_expand = tf.expand_dims(output_concat, 1)
   
    # project output: BxTx1x2H -> BxTx1xC -> BxTxC
    output_conv = tf.layers.conv2d(output_expand, filters=128, kernel_size=1,
                             strides=1, padding='SAME',
                             activation=tf.nn.relu, name="rnn_conv")
    
    #kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, 80], stddev=0.1))
    #rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
    rnn_output = tf.squeeze(output_conv, axis=[1])
    '''
    #rnn_output = tf.expand_dims(output_concat, 1)
    #rnn_output = output_concat
    rnn_output = tf.concat([fw, bw], 2)
    #print(rnn_input, '\n', fw, '\n', bw, '\n', fw_s, '\n', bw_s, '\n', rnn_output, '\n', tf.concat([fw, bw], 2))

    return rnn_output

# rnn_output is in shape of [B*80, 18, 256]   


# In[ ]:


# rnn_output_hor is in shape of [B*80, 18, 256]   

def build_rnn_ver(rnn_output_hor):
    # rnn_intput is in shape of [B*80, 18, 256]
    rnn_input = tf.reshape(rnn_output_hor, [-1, 80, rnn_output_hor.shape[1]*rnn_output_hor.shape[2]]) 
        
    n_unit = 128
    cells = [tf.contrib.rnn.LSTMCell(num_units=n_unit) for _ in range(3)] # 2 layers

    # stack basic cells
    stacked = tf.contrib.rnn.MultiRNNCell(cells)

    # bidirectional RNN
    # BxTxF -> BxTx2H
    ((fw, bw), (fw_s, bw_s)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, 
                                                               inputs=rnn_input, dtype=rnn_input.dtype, 
                                                               scope = 'rnn_ver')

    #####output_concat = tf.concat([fw_s[-1][1], bw_s[-1][1]], 1)
    
    '''
    # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
    output_concat = tf.concat([fw, bw], 2)
    output_expand = tf.expand_dims(output_concat, 1)
   
    # project output: BxTx1x2H -> BxTx1xC -> BxTxC
    output_conv = tf.layers.conv2d(output_expand, filters=128, kernel_size=1,
                             strides=1, padding='SAME',
                             activation=tf.nn.relu, name="rnn_conv")
    
    #kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, 80], stddev=0.1))
    #rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
    rnn_output = tf.squeeze(output_conv, axis=[1])
    '''
    #rnn_output = tf.expand_dims(output_concat, 1)
    #rnn_output = output_concat
    rnn_output = tf.concat([fw, bw], 2)
    #print(rnn_input, '\n', fw, '\n', bw, '\n', fw_s, '\n', bw_s, '\n', rnn_output, '\n', tf.concat([fw, bw], 2))

    return rnn_output

# rnn_output is in shape of [B, 80, 256] 


# In[ ]:


# rnn_output_ver is in shape of [B, 80, 256]  
def fully_connect(rnn_output_ver):
    
    out1 = tf.layers.dense(rnn_output_ver, 128, activation=tf.nn.relu, name = 'out1')
    out1_drop = tf.layers.dropout(out1, dropout_rate, training=training)
    out2 = tf.layers.dense(out1_drop, 32, activation=tf.nn.relu, name = 'out2')
    out2_drop = tf.layers.dropout(out2, dropout_rate, training=training)
    logits = tf.layers.dense(out2_drop, 2, name = 'logits')
   
    return logits

# logits is in shape of [B, 80, 2]


# In[ ]:


def evaluate(X_test, y_test, test_batch = 20):
    num_examples = len(X_test)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, test_batch):
        batch_x, batch_y = X_test[offset:offset+test_batch], y_test[offset:offset+test_batch]
        acc_val = sess.run(accuracy, feed_dict={X: batch_x, y: batch_y})
        total_accuracy += (acc_val * len(batch_x))
    return total_accuracy / num_examples


# In[ ]:





# In[ ]:


def construct_graph():
    reset_graph() 
    height = 10#8
    width = 600#640
    channels = 1
    n_steps = 80

    X = tf.placeholder(tf.float32, shape=[None, n_steps, height, width, channels], name="X") 
    y = tf.placeholder(tf.int64, [None, n_steps, 1])
    training = tf.placeholder_with_default(False, shape=[], name='training')
    dropout_rate = 0.5
    learning_rate = 1e-3

    cnn_out = build_cnn(X)
    rnn_hor = build_rnn_hor(cnn_out)
    rnn_ver = build_rnn_ver(rnn_hor)
    logits = fully_connect(rnn_ver)


    # now, logits is in shape = [B, 80, 2] 
    # now, y is in shape = [B, 80, 1] 

    with tf.name_scope("train"):
        y_reshape = tf.reshape(y, [-1])
        logits_reshape = tf.reshape(logits, [-1, 2])
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_reshape, logits=logits_reshape)                                                
        loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits_reshape, y_reshape, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()


# In[ ]:


def execute_graph():
    n_epochs = 50
    batch_size = 16
    best_acc = 0

    with tf.Session() as sess:
        init.run()
        #saver.restore(sess,  "./saved_model/test")
        for epoch in range(n_epochs):
            t0 = timeit.default_timer()
            for iteration in range(len(X_train)//batch_size):
                #X_batch, y_batch, _ = prepare_batch(data_train, imgpath, num = batch_size, is_train = True)
                idx = np.random.choice(len(X_train), batch_size, replace=False)
                X_batch, y_batch = X_train[idx], y_train[idx]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            #acc_test = accuracy.eval(feed_dict={X: X_test[:10], y: y_test[:10]})
            acc_test = evaluate(X_test, y_test, test_batch = 10)

            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            print(epoch, 'Running time:', timeit.default_timer() - t0)

            if acc_test > best_acc:
                best_acc = acc_test
                save_path = saver.save(sess, "./saved_model/test_2")


# In[ ]:





# In[ ]:





# In[ ]:


def prepare_data(imgsamples, ):
    imgpath = './train_images/'
imgfiles = os.listdir(imgpath)
imgsamples = [k for k in imgfiles if k.endswith('.png')]

with open('./annotate.txt') as f:
    labels = f.readlines()
    
print(len(imgsamples), len(labels))

img_train, img_test = get_train_test(imgsamples, train_ratio=0.95)
data_train = get_data(img_train, labels)
data_test = get_data(img_test, labels)

## caution about data size
X_train, y_train, _ = prepare_batch(data_train, imgpath, is_train = False)
X_test, y_test, _ = prepare_batch(data_test, imgpath, is_train = False)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[ ]:


def train():
    
    imgpath = './train_images/'
    imgfiles = os.listdir(imgpath)
    imgsamples = [k for k in imgfiles if k.endswith('.png')]

    with open('./annotate.txt') as f:
        labels = f.readlines()

    img_train, img_test = get_train_test(imgsamples, train_ratio=0.95)
    data_train = get_data(img_train, labels)
    data_test = get_data(img_test, labels)

    ## caution about data size
    X_train, y_train, _ = prepare_batch(data_train, imgpath, is_train = False)
    X_test, y_test, _ = prepare_batch(data_test, imgpath, is_train = False)
    
    
    construct_graph()
    execute_graph()
    


# In[ ]:





# In[ ]:





# In[ ]:


if __name__ == '__main__':
    train()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




