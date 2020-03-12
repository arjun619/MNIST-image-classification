#!/usr/bin/env python
# coding: utf-8

# In[150]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()


# In[151]:


img_height=img_width=28                                                #height and width of image are 28x28
img_size_flat=img_height*img_width                                     #total pixels under consideration are 784
n_classes=10                                                           #total 10 classes that is the total output numbers


# In[152]:


def load_data(mode='train'):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
    
        x_train,y_train,x_validate,y_validate=mnist.train.images,mnist.train.labels,mnist.validation.images,mnist.validation.labels
    
        x_test,y_test=mnist.test.images,mnist.test.labels
    return x_test,y_test


# In[153]:


def randomize(x,y):
    permutation=np.random.permutation(y.shape[0])
    shuffled_x=x[permutation, :]
    shuffled_y=y[permutation]
    return shuffled_x,shuffled_y


# In[154]:


def get_next_batch(x,y,start,end):
    x_batch=x[start:end]
    y_batch=y[start:end]
    return x_batch,y_batch


# In[155]:


x_train,y_train,x_valid,y_valid=load_data(mode='train')


# In[156]:


x_train[0];


# In[157]:


#after loading the dataset we decide for the hyperparameters
#these including the use of number of epochs, mini batch size , regularization parameters,learning rate alpha etc
epochs=10                                                    
batch_size=100
display_freq=100
learning_rate=0.001
h1=200                                 #number of units in first hidden layer


# In[158]:


def weight_variable(name,shape):
    initer=tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('w_'+name,dtype=tf.float32,shape=shape,initializer=initer)
    


# In[159]:


def bias_variable(name,shape):
    initial=tf.constant(0.,shape=shape,dtype=tf.float32)
    return tf.get_variable('b_'+name,dtype=tf.float32,initializer=initial)


# In[160]:


def fc_layer(x,num_units,name,use_relu=True):
    #x is the input from the previous layer
    #num_units is the number of inputs in the previous layer
    #name of the layer
    #use activation function as relu if none is specified
    in_dim=x.get_shape()[1]
    w=weight_variable(name,[in_dim,num_units])
    b=bias_variable(name,[num_units])
    layer=tf.matmul(x,w)
    layer+=b
    if use_relu=True:
        layer=tf.nn.relu(layer)
    return layer


# In[161]:


#make placeholder for holding the pixels information and the output class
x=tf.placeholder(tf.float32,shape=[None,img_size_flat],name='x')
y=tf.placeholder(tf.float32,shape=[None,n_classes],name='y')


# In[162]:



fc1=fc_layer(x,h1,'FC6',use_relu=True)                                          #layer 1 of the network
output_logits=fc_layer(fc1,n_classes,'OUT',use_relu=False)                      #layer 2 of the network. in total 1 hidden layer


# In[163]:


#set the loss function,optimizer etc

#argmax returns the index of the element which has the maximum value
cls_prediction=tf.argmax(output_logits,axis=1,name='predictions')

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output_logits),name="loss")
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,name="adam-op").minimize(loss)
correct_prediction=tf.equal(tf.argmax(output_logits,1),tf.argmax(y,1),name="correct_prediction")
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")


# In[164]:


init=tf.global_variables_initializer()


# In[165]:


#run the model
sess=tf.InteractiveSession()
sess.run(init)#initialize the variables
global_count=0
#number of training iterations per epoch
num_tr_iter=int(len(y_train)/(batch_size))
for epoch in range((epochs)):
    x_train,y_train=randomize(x_train,y_train)
    for iteration in range((num_tr_iter)):
        global_count+=1
        start=iteration*batch_size
        end=(iteration+1)*batch_size
        x_batch,y_batch=get_next_batch(x_train,y_train,start,end)
        #run the optimization procedure
        feed_dict_batch={x:x_batch,y:y_batch}
        sess.run(optimizer,feed_dict=feed_dict_batch)
            
        if iteration%display_freq==0:
            loss_batch,acc_batch=sess.run([loss,accuracy],feed_dict=feed_dict_batch)
            print("loss ",loss_batch," at iteration ",iteration," accuracy ",acc_batch)
    feed_dict_valid={x:x_valid[:1000],y:y_valid[:1000]}
    loss_valid,acc_valid=sess.run([loss,accuracy],feed_dict=feed_dict_valid)
    print("loss ",loss_valid," at iteration ",iteration," accuracy ",acc_valid)
        


# In[170]:


#data visualization
def plot_images(images,cls_true,cls_pred=None,title=None):
    fig,axes=plt.subplots(3,3,figsize=(9,9))
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    for i,ax in enumerate(axes.flat):
        #plot image
        ax.imshow(images[i].reshape(28,28),cmap='binary')
        #show true and predicted labels
        if cls_pred is None:
            ax_title="True {0}".format(cls_true[i])
        else:
            ax_title="True {0} pred {1}".format(cls_true[i],cls_pred[i])
        ax.set_title(ax_title)
        
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        plt.suptitle(title, size=20)
    plt.show(block=False)
        


# In[171]:


def plot_example_errors(images, cls_true, cls_pred, title=None):
    """
    Function for plotting examples of images that have been mis-classified
    :param images: array of all images, (#imgs, img_h*img_w)
    :param cls_true: corresponding true labels, (#imgs,)
    :param cls_pred: corresponding predicted labels, (#imgs,)
    """
    # Negate the boolean array.
    incorrect = np.logical_not(np.equal(cls_pred, cls_true))

    # Get the images from the test-set that have been
    # incorrectly classified.
    incorrect_images = images[incorrect]

    # Get the true and predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    cls_true = cls_true[incorrect]

    # Plot the first 9 images.
    plot_images(images=incorrect_images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9],
                title=title)
 


# In[172]:


# Test the network after training
x_test, y_test = load_data(mode='test')
feed_dict_test = {x: x_test[:1000], y: y_test[:1000]}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
print('---------------------------------------------------------')

# Plot some of the correct and misclassified examples
cls_pred = sess.run(cls_prediction, feed_dict=feed_dict_test)
cls_true = np.argmax(y_test[:1000], axis=1)
plot_images(x_test, cls_true, cls_pred, title='Correct Examples')
plot_example_errors(x_test[:1000], cls_true, cls_pred, title='Misclassified Examples')
plt.show()


# In[ ]:




