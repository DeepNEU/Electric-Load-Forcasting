import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend as K
from keras import optimizers
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Activation, Lambda
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
import keras.layers

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import grid_rnn
import numpy as np
import pandas as pd
import random
import eleceval
from sklearn import preprocessing
from tensorflow.python.ops import array_ops

import keras

def pinball_loss(y_true, y_pred):
    '''
    The formula is shown in the picture named formula of Pinball-loss
    r :   Neighborhood range   named ϵ in picture
    location :  record the comparision between observed value and predicted value
    re ： tensor of r
    e  ： Absolute value of subtraction between observed value and predicted value
    position ：record the comparision between e and re
    h1 : first part of formula H 
    h2 :second part of formula H 
    h : result of formula H
    p : value of loss function
    '''
    print('_true;')
    print(y_true.shape)
    print(y_pred.shape)
    p=0
    q1=tf.linspace(0.1,0.9,9)
    q2=1-q1
    r=0.01
    location=tf.less(y_true,y_pred)
    e = tf.abs(y_true - y_pred)
    
    re=tf.ones_like(e)*r
    position=tf.less(e,re)
    
    h1=tf.square(e)/(2*r)
    h2=e-r/2
    h=tf.where(position,h1,h2)
    
    p1=tf.multiply(q2,h)
    p2=tf.multiply(q1,h)
    p=tf.reduce_mean(tf.where(location,p1,p2),axis=0)
    p=tf.reduce_mean(p)
    
    return p

def pinball_score(true, pred, number_q):
    q=np.linspace(1/(number_q+1),(1-(1/(number_q+1))),number_q)
    loss = np.where(np.less(true,pred), (1-q)*(np.abs(true-pred)), q*(np.abs(true-pred)))
    return np.mean(loss)

def winkler_score(true, lower,upper, PI):
    score_tmp = np.where(np.less(upper,true),2*(true-upper)/(1-PI),np.zeros_like(true))
    score = np.where(np.less(true,lower),2*(lower-true)/(1-PI),score_tmp)
    return np.mean(score+upper-lower)

def PCIP(true, lower,upper, PI):
    score_tmp = np.where(np.less(upper,true),np.zeros_like(true),np.ones_like(true))
    score = np.where(np.less(true,lower),np.zeros_like(true),score_tmp)
    return np.mean(score)

def ECR(lower,upper):
    #return np.mean(upper-lower)
    return (np.mean(upper-lower))

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return keras.layers.multiply([tanh_out, sigm_out])


def residual_block(x, s, i, activation, nb_filters, kernel_size):
    original_x = x

    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=2 ** i,padding='same',
                  name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(0.1)(x)

    # 1x1 conv.
    x = Convolution1D(nb_filters, 1, padding='same')(x)
    res_x = keras.layers.add([original_x, x])
    return res_x, x


def x_c_multiply(x,cw,max_len,number_cluster):
    cw = tf.expand_dims(cw,-1)
    cw = tf.tile(cw,[1,1,1,max_len])#(batch,num_residual,number_cluster,max_len)
    cw = tf.transpose(cw,[0,1,3,2])#(batch,num_residual,max_len,number_cluster)
    x = tf.expand_dims(x,-1)
    x = tf.tile(x,[1,1,1,number_cluster])#(batch,num_residual,max_len,number_cluster)
    x = tf.multiply([x, cw])#(batch,num_residual,max_len,number_cluster)
    x = tf.transpose(x,[0,2,3,1])#(batch,max_len,number_cluster,num_residual)
    return tf.reduce_mean(x,axis=-1, keep_dims=False)#(batch,max_len,number_cluster)
    
def clustering(x,number_cluster,max_len):
    
    cw = Dense(100,activation="sigmoid")(x)
    cw = Dense(48,activation="relu")(cw)
    
    dname="Dense"+str(number_cluster)+"-1"
    cw = Dense(number_cluster,activation="softmax",name=dname)(x)
    
    
    cw = Lambda(lambda tt: tf.expand_dims(tt,-1))(cw)
    cw = Lambda(lambda tt: tf.tile(tt,[1,1,1,max_len]))(cw)
    cw = Lambda(lambda tt: tf.transpose(tt,[0,1,3,2]))(cw)
    
    x = Lambda(lambda tt: tf.expand_dims(tt,-1))(x)
    x = Lambda(lambda tt: tf.tile(tt,[1,1,1,number_cluster]))(x)
    
    x = keras.layers.multiply([x, cw])
    x = Lambda(lambda tt: tf.transpose(tt,[0,2,3,1]))(x)
    x = Lambda(lambda tt: tf.reduce_mean(tt,axis=-1, keep_dims=False))(x)
    return x


def dilated_tcn(num_feat, num_classes, nb_filters,
                kernel_size, dilatations, nb_stacks, max_len,
                activation='wavenet', use_skip_connections=True,
                return_param_str=False, output_slice_index=None,
                regression=False,number_cluster = 3):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """
    input_layer = Input(name='input_layer', shape=(max_len, num_feat))
    x = input_layer
    
    x_all = Lambda(lambda tt: tf.reduce_mean(tt,axis=-1, keep_dims=True))(x)
    
    x = Lambda(lambda tt: tf.transpose(tt,[0,2,1]))(x)#(batch,max_len,num_residual)->(batch,num_residual,max_len)
    
    ensemble=[2,3,4,5,6,7,8,10,20]
    for i in ensemble:
        tmp = clustering(x,i,max_len)
        x_all = keras.layers.concatenate([x_all, tmp])

    
    
    x = Convolution1D(nb_filters, kernel_size, padding='same', name='initial_conv_2')(x_all)
    

    skip_connections = []
    for s in range(nb_stacks):
        for i in dilatations:
            x, skip_out = residual_block(x, s, i, activation, nb_filters, kernel_size)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = keras.layers.add(skip_connections)
    x = Activation('relu')(x)

    if output_slice_index is not None:  # can test with 0 or -1.
        if output_slice_index == 'last':
            output_slice_index = -1
        if output_slice_index == 'first':
            output_slice_index = 0
        print('first:x.shape=', x.shape)
        x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)

    

    if not regression:
        # classification
        x = Dense(9)(x)
        
        x = Activation('softmax', name='output_softmax')(x)
        
        output_layer = x
        print(f'model.x = {input_layer.shape}')
        print(f'model.y = {output_layer.shape}')
        model = Model(input_layer, output_layer)

        adam = optimizers.Adam(lr=0.005, clipnorm=1.)
        model.compile(adam, loss=pinball_loss, metrics=['accuracy'])
        print('Adam with norm clipping.')
    else:
        # regression

        x = Dense(9)(x)
        #print('2.x.shape=', x.shape)
        output_layer = Activation('linear', name='output_dense')(x)
        print(f'model.x = {input_layer.shape}')
        print(f'model.y = {output_layer.shape}')
        model = Model(input_layer, output_layer)
        
        
        
        adam = optimizers.Adam(lr=0.02, clipnorm=1.,decay=0.1,amsgrad=True)
        #sgd = optimizers.SGD(lr=0.005, momentum=0.9, decay=0.0, nesterov=False)
        model.compile(adam, loss=pinball_loss)

    if return_param_str:
        param_str = 'D-TCN_C{}_B{}_L{}'.format(2, nb_stacks, dilatations)
        return model, param_str
    else:
        return model






predictperiod = '6h' #15m:15分钟，6h：6小时，1d：天
modeltype = 'TCN' #LSTM,GRU,pLSTM,gridLSTM

summary_dir = "/media/dzf/data/data/MSFNN_train_result/0_test/"+modeltype+predictperiod
MODEL_SAVE_PATH = "/media/dzf/data/data/MSFNN_train_result/0_test/"+modeltype+predictperiod
MODEL_NAME = "model.ckpt"

#learning_rate = 0.001
training_iters = 40001
batch_size = 2400
regularization_rate = 0.0001

data_size = 2000-48
train_data = int(data_size*0.8)
val_data = int(data_size*0.1)


n_input = 5
#用前10个数据预测下一个,第batch_size个数据，n_step个为一组，一个n_input个特征
n_steps = 48
n_hidden = 400
n_class = 1 
n_layers = 18
num_epochs=5000
data_path = "929customerload.csv"    
period_step = 24
period_day = 7


def dofile(filename,datasize):
    df = pd.read_csv(filename, index_col=0)
    X = [] ; Y = []
    for i in range(datasize-n_steps):
        x = df.drop(columns=['sum']).loc[i:i+n_steps-1].values.tolist()
        y = df.loc[i+n_steps,['sum']].tolist()
        X.append(x)
        yy = y*9
        Y.append(yy)
    return X,Y

#载入训练数据
x_raw,y_raw = dofile(data_path, data_size)

xtrain = x_raw[0:train_data]
ytrain = y_raw[0:train_data]
x_train=np.array(xtrain)*10.0
y_train=np.array(ytrain).reshape(-1,9)

xval = x_raw[train_data:train_data+val_data]
yval = y_raw[train_data:train_data+val_data]
x_val=np.array(xval)*10.0
y_val = np.array(yval).reshape(-1,9)

xtest = x_raw[train_data+val_data:data_size-n_steps]
ytest = y_raw[train_data+val_data:data_size-n_steps]
x_test=np.array(xtest)*10.0
y_test=np.array(ytest).reshape(-1,9)
    




class PrintSomeValues(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.mape_flag = 100.0
        self.pinball_flag=100
    
    
    def on_epoch_begin(self, epoch, logs={}):
        #print(f'x_test[0:1] = {x_test[0:1]}.')
        #print(f'y_test[0:1] = {y_test[0:1]}.')
        #print(f'pred = {self.model.predict(x_test[0:1])}.')
        lr = K.get_value(model.optimizer.lr)
        print("current learning rate is {}".format(lr))
        pred = model.predict(x_test)

        predict_all = pred.flatten()
        truth_all = y_test.flatten()
        
        mape = eleceval.calcMAPE(predict_all,truth_all)
        mae = eleceval.calcMAE(predict_all,truth_all) 
        mse = eleceval.calcMSE(predict_all,truth_all)
        rmse = eleceval.calcRMSE(predict_all,truth_all) 
        r_2 = eleceval.r2(predict_all,truth_all)
        pinball=pinball_score(y_test,pred,9)
        prediction=pred
        truth=y_test[:,0]
        winkler40 = winkler_score(truth, prediction[:,2],prediction[:,6], 0.4)
        winkler80 = winkler_score(truth, prediction[:,0],prediction[:,8], 0.8)
        pcip40 = PCIP(truth, prediction[:,2],prediction[:,6], 0.4)
        pcip80 = PCIP(truth, prediction[:,0],prediction[:,8], 0.8)
        ace40 = pcip40 - 0.4
        ace80 = pcip80 - 0.8
        ecr40 = ECR(prediction[:,2],prediction[:,6])/(np.max(truth)-np.min(truth))
        ecr80 = ECR(prediction[:,0],prediction[:,8])/(np.max(truth)-np.min(truth))
        print("After %d training step(s),"
              "on test data MAPE = %.4f,MAE = %.4f,MSE = %.4f,RMSE = %.4f,R2 = %.4f,piball_loss = %.4f,winkler40 = %.4f,winkler80 = %.4f,pcip40 = %.4f,pcip80 = %.4f,ace40 = %.4f,ace80 = %.4f,ecr40 = %.4f,ecr80 = %.4f"\
              % (epoch, mape,mae,mse,rmse,r_2,pinball,winkler40,winkler80,pcip40,pcip80,ace40,ace80,ecr40,ecr80))
        
        if pinball < self.pinball_flag:
            self.pinball_flag = pinball
            #两个ndarray列合并
            #y_con = np.concatenate((truth_all, predict_all), axis=1)
            truth_all_reshape=np.reshape(truth_all,[-1,1])
            predict_all_reshape=np.reshape(predict_all,[-1,1])
            y_con = np.concatenate((truth_all_reshape, predict_all_reshape), axis=1)
            #输出真实值和预测值
            y_out = pd.DataFrame(y_con, columns=["true_data","pre_data"])
            y_out.to_csv('./result_929_3cluster/steps=%d-MAPE=%.4f,MAE = %.4f,MSE = %.4f,RMSE = %.4f,R2 = %.4f,piball_loss = %.4f,winkler40 = %.4f,winkler80 = %.4f,pcip40 = %.4f,pcip80 = %.4f,ace40 = %.4f,ace80 = %.4f,ecr40 = %.4f,ecr80 = %.4f.csv'\
                        % (epoch,mape,mae,mse,rmse,r_2,pinball,winkler40,winkler80,pcip40,pcip80,ace40,ace80,ecr40,ecr80))



#def run_task():

model, param_str = dilated_tcn(output_slice_index='last',
                               num_feat=x_train.shape[2],
                               num_classes=0,
                               nb_filters=12,
                               kernel_size=3,
                               dilatations=[0, 1, 2, 3],
                               nb_stacks=1,
                               max_len=x_train.shape[1],
                               activation='norm_relu',
                               use_skip_connections=False,
                               return_param_str=True,
                               regression=True,
                               number_cluster = 4)
#1.kernel_size=8,dropout=0.05,learning_rate=0.002,batch_size=128,dilatations=[1, 2, 4, 8]

print(f'x_train.shape = {x_train.shape}')
print(f'y_train.shape = {y_train.shape}')

psv = PrintSomeValues()

# Using sparse softmax.
# http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
model.summary()


reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=15, mode='min')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=2,mode='min')

#for i in range(1000):
    #callbacks=[psv]
model.fit(x_train, y_train, 
          validation_data=(x_val, y_val),
          epochs=5000, 
          batch_size=32,
          initial_epoch=0,
          callbacks=[early_stopping,reduce_lr, psv]
         )

#if __name__ == '__main__':
#    run_task()
