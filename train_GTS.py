# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
import random
from skimage import transform
from sklearn import preprocessing
from sklearn.utils import shuffle

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrainData(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        gtReader.next() # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels.append(int(row[7])) # the 8th column is the label
        gtFile.close()
    return images, labels

def readTestData(csv_dir):
    images = []
    labels = []
    prefix = csv_dir + '/'
    csv_path = prefix + 'GT-final_test.csv'
    gtFile = open(csv_path)
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.next() # skip header
    
    for row in gtReader:
        images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
        labels.append(int(row[7])) # the 8th column is the label
    gtFile.close()
    return images, labels

#show samples images for each class
def sample_im_show(images, labels, uni_labels):
    plt.figure(figsize=(20, 20))
    i = 1
    for lb in uni_labels:
        #select the 1st image of each class
        im = images[labels.index(lb)]
        plt.subplot(8,8,i)
        i += 1
        plt.axis('off')
        plt.title("Class {0}-{1}".format(lb, labels.count(lb)))
        plt.imshow(im)
    plt.show()

#image normalization
def image_normalization(images):
    images = np.array(images)
    images = images.astype(np.float32)
    #preprocessing: normalization
    images_norm = []
    for im in images:
        for band in range(0,3):
            im[:,:,band] = preprocessing.scale(im[:,:,band])
        images_norm.append(im) 
    return images_norm

#initilize weight
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    
#initilize bias
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
    
#create CNN
def create_cnn_layer(input, 
                     num_input_channels, 
                     conv_filter_size, 
                     num_filters):
    #initialize weights
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])       
    #initialize biases
    biases = create_biases(num_filters)    
    
    #create the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1,1,1,1],
                         padding='SAME')
    layer += biases
                    
    #max-pooling
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1,2,2,1],
                           strides=[1,2,2,1],
                           padding='SAME')
    
    #activation func ReLU
    layer = tf.nn.relu(layer)

    return layer      

#Flattening layer
def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer

#fully connected layer
def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])  
    biases = create_biases(num_outputs)
    
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer

#get batch data of size: size_batch from training data
def batch_train(i_iteration, image_train, label_train, size_batch):
    num_start = i_iteration*size_batch
    im_train_batch = image_train[num_start : (num_start + size_batch)]
    lb_train_batch = label_train[num_start : (num_start + size_batch)]
    return im_train_batch, lb_train_batch

#randomly select a batch of samples from a validation data
def batch_validation(image_validation, label_validation, size_batch):
    batchIdx = random.sample(range(len(image_validation)), size_batch)
    im_val_batch = [image_validation[i] for i in batchIdx]
    lb_val_batch = [label_validation[i] for i in batchIdx]
    
    return im_val_batch, lb_val_batch

"""
Step 1: Load data
"""   
print("Step 1: Load data")
#load training data
im_train, lb_train = readTrainData('./Final_Training/Images')
#randomize the order of the data
im_train, lb_train = shuffle(im_train, lb_train)
#load testing data
#im_test, lb_test = readTestData("./Final_Test/Images")

#get unique labels
uni_label = set(lb_train)
num_classes = len(uni_label)
#display samples images for each class
sample_im_show(im_train, lb_train, uni_label)

"""
Step 2: Data preprocessing
"""
print("Step 2: Data preprocessing")
#resize the images to be consistent
im_size = 32
im_train = [transform.resize(im, (im_size, im_size)) for im in im_train]
im_train = np.array(im_train)
im_train = im_train.astype(np.float32)
lb_train = np.array(lb_train)
lb_train = lb_train.astype(np.int32)
#im_test = [transform.resize(im, (im_size, im_size)) for im in im_test]

"""
#load testing data
"""
im_size = 32
im_test, lb_test = readTestData("./Final_Test/Images")
im_test = [transform.resize(im, (im_size, im_size)) for im in im_test]
im_test = np.array(im_test)
im_test = im_test.astype(np.float32)
lb_test = np.array(lb_test)
lb_test = lb_test.astype(np.int32)


#image normalization, data type changed to float32
#im_train = image_normalization(im_train)
#im_test = image_normalization(im_test)
#change color image to grayscale
#im_train = np.array(im_train)
#im_train = [rgb2gray(im) for im in im_train]
num_im_channels = 3

"""
Step 3: Divide im_train data into training part and validation part
"""
print("Step 3: Divide im_train data into training part and validation part")

trn_proportion = 0.8
sz_im_train = len(im_train)
#number of images in im_train for training
sz_imtrain_trn = int(sz_im_train * trn_proportion)
#number of images in im_train for validation
sz_imtrain_val = sz_im_train - sz_imtrain_trn
#training data
imtrain_trn = im_train[0:sz_imtrain_trn]
lbtrain_trn = lb_train[0:sz_imtrain_trn]
#validation data
imtrain_val = im_train[sz_imtrain_trn:sz_im_train]
lbtrain_val = lb_train[sz_imtrain_trn:sz_im_train]

"""
Step 4: Design CNN 
"""
print("Step 4: Design CNN")

#create a graph to hold the CNN model
graph = tf.Graph()
#create CNN model in the graph
with graph.as_default():
    #placeholders for inputs and labels
    x = tf.placeholder(tf.float32, shape=[None, im_size, im_size, num_im_channels], name='x')
    y_true = tf.placeholder(tf.int32, shape=[None], name='y_true')
    
    #design the network
    #Network graph params
    filter_size_conv1 = 3
    num_filters_conv1 = 32
    
    filter_size_conv2 = 3
    num_filters_conv2 = 32
    
    filter_size_conv3 = 3
    num_filters_conv3 = 64
    
    fc_layer_size = 128
    
    #create each layer
    layer_conv1 = create_cnn_layer(input=x,
                                   num_input_channels=num_im_channels, 
                                   conv_filter_size=filter_size_conv1, 
                                   num_filters=num_filters_conv1)
                                             
    layer_conv2 = create_cnn_layer(input=layer_conv1,
                                   num_input_channels=num_filters_conv1, 
                                   conv_filter_size=filter_size_conv2, 
                                   num_filters=num_filters_conv2)                                       
    
    layer_conv3 = create_cnn_layer(input=layer_conv2,
                                   num_input_channels=num_filters_conv2, 
                                   conv_filter_size=filter_size_conv3, 
                                   num_filters=num_filters_conv3) 
                                             
    layer_flat = create_flatten_layer(layer_conv3) 
    
    layer_fc1 = create_fc_layer(input=layer_flat,
                        num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                        num_outputs=fc_layer_size,
                        use_relu=True)   
    
    layer_fc2 = create_fc_layer(input=layer_fc1,
                        num_inputs=fc_layer_size,
                        num_outputs=num_classes,
                        use_relu=False)    
    
    #prediction
#    y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
    
    #define loss, optimizer
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_true,
                                                                   logits = layer_fc2)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    
    #get the prediceted class
    y_pred = tf.argmax(layer_fc2, dimension=1, output_type = tf.int32, name='y_pred')
    
    #compute correct prediction
    correct_pred = tf.equal(y_pred, y_true, name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    #saver = tf.train.Saver()
    init = tf.global_variables_initializer()

"""
Step 5: Train CNN 
"""
print("Step 5: Train CNN")
sess = tf.Session(graph=graph)
#initialize variables
_ = sess.run(init)

num_epochs = 20
size_batch = 64
num_iterations = sz_imtrain_trn / size_batch

test_acc = []
trn_acc = []
val_acc = []



for ep in range(num_epochs):
    #shuffle data
#    imtrain_trn, lbtrain_trn = shuffle(imtrain_trn, lbtrain_trn)
#    imtrain_val, lbtrain_val = shuffle(imtrain_val, lbtrain_val)
    
    for it in range(num_iterations):
        #get batch data of size: size_batch from training data
        im_trn_batch, lb_trn_batch = batch_train(it, imtrain_trn, lbtrain_trn, size_batch)
        #randomly select a batch of data from validation data
        im_val_batch, lb_val_batch = batch_validation(imtrain_val, lbtrain_val, size_batch)
    
        Feed_Dict_Trn = {x: im_trn_batch, y_true: lb_trn_batch} 
        Feed_Dict_Val = {x: im_val_batch, y_true: lb_val_batch}
        
        #train CNN
        sess.run(train_op, feed_dict = Feed_Dict_Trn)
        
        if it == num_iterations-1:
            #at the end of this epoch
            #compute accuracy and loss
            trn_acc.append(sess.run(accuracy, feed_dict = Feed_Dict_Trn))
            val_acc.append(sess.run(accuracy, feed_dict = Feed_Dict_Val))
            val_loss= sess.run(loss, feed_dict = Feed_Dict_Val)
            #save the network
            #saver.save(sess, "./Network_Save/netowrk")    
            #print out the results
            msg = "Epoch: {0}...Train acc: {1:>6.1%}...Validation acc: {2:>6.1%}...Validation loss: {3:.3f}"
            print(msg.format(ep+1, trn_acc[ep], val_acc[ep], val_loss))
            
            #calculate accuracy on testing data
            test_acc.append(sess.run(accuracy, feed_dict = {x:im_test, y_true:lb_test}))
            print("Testing Acc: {:>6.1%}".format(test_acc[ep]))
            
#plot 
test_acc = np.array(test_acc)
trn_acc = np.array(trn_acc)           
val_acc = np.array(val_acc)

np.savetxt('test_acc.txt', test_acc)
np.savetxt('trn_acc.txt', trn_acc)
np.savetxt('val_acc.txt', val_acc)
                        
plt.plot(test_acc)
plt.show()

sess.close()
