"""
call tensorflow to train and predict
this file is modify base on 
https://github.com/adventuresinML/adventures-in-ml-code/blob/master/tensor_flow_tutorial.py

 Copyright 2019: zhao shi rong   shxzhaosr@163.com

All Rights Reserved.


 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *

"""





import tensorflow as tf
import numpy as np
import pricedb
from tensorflow.examples.tutorials.mnist import input_data


def run_simple_graph():
    # first, create a TensorFlow constant
    const = tf.constant(2.0, name="const")

    # create TensorFlow variables
    b = tf.Variable(2.0, name='b')
    c = tf.Variable(1.0, name='c')

    # now create some operations
    d = tf.add(b, c, name='d')
    e = tf.add(c, 2, name='e')
    a = tf.multiply(d, e, name='a')

    # setup the variable initialisation
    init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # compute the output of the graph
        a_out = sess.run(a)
        print("Variable a is {}".format(a_out))


def run_simple_graph_multiple():
    # first, create a TensorFlow constant
    const = tf.constant(2.0, name="const")

    # create TensorFlow variables
    b = tf.placeholder(tf.float32, [None, 1], name='b')
    c = tf.Variable(1.0, name='c')

    # now create some operations
    d = tf.add(b, c, name='d')
    e = tf.add(c, 2, name='e')
    a = tf.multiply(d, e, name='a')

    # setup the variable initialisation
    init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # compute the output of the graph
        a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
        print("Variable a is {}".format(a_out))


def simple_with_tensor_board():
    const = tf.constant(2.0, name="const")

    # Create TensorFlow variables
    b = tf.Variable(2.0, name='b')
    c = tf.Variable(1.0, name='c')

    # now create some operations
    d = tf.add(b, c, name='d')
    e = tf.add(c, const, name='e')
    a = tf.multiply(d, e, name='a')

    # setup the variable initialisation
    init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # compute the output of the graph
        a_out = sess.run(a)
        print("Variable a is {}".format(a_out))
        train_writer = tf.summary.FileWriter('C:\\Users\\Andy\\PycharmProjects')
        train_writer.add_graph(sess.graph)


import nntrain

def nn_example():
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Python optimisation variables
    learning_rate = 0.002
    epochs = 40000
    batch_size = 100

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 152],name='x')
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 1])

    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([152, 300], stddev=0.03), name='W1')
    b1 = tf.Variable(tf.random_normal([300]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    W2 = tf.Variable(tf.random_normal([300, 1], stddev=0.03), name='W2')
    b2 = tf.Variable(tf.random_normal([1]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, W1), b1)
    #hidden_out = tf.nn.relu(hidden_out)
    hidden_out = tf.nn.tanh(hidden_out)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    #y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
    y_ = tf.nn.tanh(tf.add(tf.matmul(hidden_out, W2), b2))

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    #cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
    #                                              + (1 - y) * tf.log(1 - y_clipped), axis=1))

    #target= tf.reduce_mean(tf.reduce_sum((y-y_)*(y-y_), axis=1))
    target= tf.reduce_mean(tf.reduce_sum(tf.abs(y-y_), axis=1))
    #target= tf.reduce_mean(tf.reduce_sum((y-y_), axis=1))
    # add an optimiser
    #optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(target)
    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # add a summary to store the accuracy
    tf.summary.scalar('accuracy', accuracy)
    predict=tf.reduce_sum(y_, axis=1,name='predict')
    ##添加 这个目标， 以计算预测利润
    
    trainnum,profitnum,randomlist,samlength,loaddict=nntrain.testData_B()

    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter('C:\\Users\\Andy\\PycharmProjects')
    # start the session
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        #total_batch = int(len(mnist.train.labels) / batch_size)
        total_batch = int(samlength / batch_size)
        for epoch in range(epochs):
            randomlist=nntrain.generate_random_sample_list(trainnum,profitnum)
            avg_cost = 0
            for i in range(total_batch):
                #batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                batch_x, batch_y = nntrain.generate_batch_sample(trainnum,profitnum,randomlist, batch_size, i)
                _, c = sess.run([optimiser, target], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
            if(epoch%100==0):
                saver.save(sess, './2017/my_test_2017model',global_step=epoch)
            #summary = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            #writer.add_summary(summary, epoch)

        print("\nTraining complete!")
        #writer.add_graph(sess.graph)
        #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
        codeResultDict, stockcodelist,inputlistnp=nntrain.computeOneDaybyNeuralNetwork( loaddict,   '2018-12-12')
        ##以后边数据训练预测前边日期， 这个要修改
        #print(sess.run(predict, feed_dict={x: inputlistnp}))
        npprelist=sess.run(predict, feed_dict={x: inputlistnp})
        print(npprelist) 
        prelist=npprelist.tolist()
        print (sorted(prelist))
        nntrain.putPredictProfitToCodeResultDict(codeResultDict,stockcodelist,prelist)
        sorted_by_value = sorted(codeResultDict.items(), key=lambda kv: kv[1]['predict'])
        #print (sorted_by_value)
        sorted_by_value.reverse()
        for i in range(0, 5):
            print (sorted_by_value[i][0])
            print (sorted_by_value[i][1]['predict'])
            print (sorted_by_value[i][1]['profit'])
            print (sorted_by_value[i][1]['stockcode'])
        #saver.save(sess, './2017/my_test_2017model_final',global_step=1000)
        saver.save(sess, './2017/my_test_2017model_final')
        
        
def   restore_model_and_predict():
    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    trainnum,profitnum,randomlist,samlength,loaddict=nntrain.testData_B()
    codeResultDict, stockcodelist,inputlistnp=nntrain.computeOneDaybyNeuralNetwork( loaddict,   '2018-12-12') 
 
    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data  
 
    graph = tf.get_default_graph()
     #w1 = graph.get_tensor_by_name("w1:0")
     #w2 = graph.get_tensor_by_name("w2:0")
     #feed_dict ={w1:13.0,w2:17.0}
  
     #Now, access the op that you want to run. 
    x=graph.get_tensor_by_name("x:0")
    predict = graph.get_tensor_by_name("predict:0")
 
    npprelist= sess.run(predict,feed_dict={x: inputlistnp})
    print(npprelist) 
    prelist=npprelist.tolist()
    print (sorted(prelist))
    nntrain.putPredictProfitToCodeResultDict(codeResultDict,stockcodelist,prelist)
    sorted_by_value = sorted(codeResultDict.items(), key=lambda kv: kv[1]['predict'])
        #print (sorted_by_value)
    sorted_by_value.reverse()
    for i in range(0, 5):
            print (sorted_by_value[i][0])
            print (sorted_by_value[i][1]['predict'])
            print (sorted_by_value[i][1]['profit'])
            print (sorted_by_value[i][1]['stockcode'])
#This will print 60 which is calculated         
  
def  computeOneDaybySession(sess,x,predict, loaddict,Date):
    codeResultDict, stockcodelist,inputlistnp=nntrain.computeOneDaybyNeuralNetwork( loaddict,   Date)
    npprelist= sess.run(predict,feed_dict={x: inputlistnp})
    print(npprelist) 
    prelist=npprelist.tolist()
    #print (sorted(prelist))
    nntrain.putPredictProfitToCodeResultDict(codeResultDict,stockcodelist,prelist)
    sorted_by_value = sorted(codeResultDict.items(), key=lambda kv: kv[1]['predict'])
        #print (sorted_by_value)
    sorted_by_value.reverse()
    totalprofit=0.0
    print ("total stocks %d" %(len(sorted_by_value)))
    for i in range(0, 5):
            print (sorted_by_value[i][0])
            print (sorted_by_value[i][1]['predict'])
            print (sorted_by_value[i][1]['profit'])
            print (sorted_by_value[i][1]['stockcode'])
            totalprofit+=sorted_by_value[i][1]['profit']
            averprofit=totalprofit/5.0
            print (averprofit)
    redict={}
    redict[Date]=averprofit
    return redict


def  restore_and_compute():
    sess=tf.Session()    
    #First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('./2017/my_test_2017model-700.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./2017'))
    trainnum,profitnum,randomlist,samlength,loaddict=nntrain.testData_B()
    tradedatelist=pricedb.buildtradedatelist(loaddict)
    graph = tf.get_default_graph()
     #w1 = graph.get_tensor_by_name("w1:0")
     #w2 = graph.get_tensor_by_name("w2:0")
     #feed_dict ={w1:13.0,w2:17.0}
  
     #Now, access the op that you want to run. 
    x=graph.get_tensor_by_name("x:0")
    predict = graph.get_tensor_by_name("predict:0")
    
    startdate='2018-01-01'
    enddate='2018-12-28'
    startidx=-1
    for idx, date in enumerate(tradedatelist):
        if(date>=startdate):
            startidx=idx
            break
    if startidx==-1:
        return 
    
    endidx=len(tradedatelist)
    for idx, date in enumerate(tradedatelist):
        if(date>enddate):
             endidx=idx-1
             break
    if endidx==-1:
        return 
    resudict={}
    for j in range(startidx, endidx):
        
           curday=tradedatelist[j]
           rt=computeOneDaybySession(sess,x,predict, loaddict,curday)
           resudict.update(rt)
    
    
    
    for key in sorted(resudict.keys()):
        print ("%s: %s" % (key, resudict[key]))
    fullprofit=0.0  
    numdays=0
    for date, profit in resudict.items():
        #print (date)
        #print (profit)
        fullprofit+=profit
        numdays+=1
    daver=fullprofit/numdays    
    print ("days ", numdays)
    print ("fullprofit ", fullprofit)
    print ("aver ", daver)


      
if __name__ == "__main__":
    #run_simple_graph()
    # run_simple_graph_multiple()
    # simple_with_tensor_board()
    #nn_example()
    #restore_model_and_predict()
    restore_and_compute()