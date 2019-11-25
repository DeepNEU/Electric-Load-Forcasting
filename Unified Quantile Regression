import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#resnet+dilate+48个小时输出
import keras.backend as K
from keras import optimizers
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Activation, Lambda
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
import keras.layers
from gated_cnn import  GatedConvBlock 
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
    conv = GatedConvBlock(Conv1D(filters=nb_filters*2, kernel_size=kernel_size,
                  dilation_rate=1,padding='same',
                  name='dilated_conv_%d_tanh_s%d' % (2 ** i, s)))(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    #x = SpatialDropout1D(0.03)(x)

    # 1x1 conv.
    x = Convolution1D(nb_filters, 1, padding='same')(x)
    
    res_x = keras.layers.add([original_x, x])
    return res_x, x


def dilated_tcn(num_feat, num_classes, nb_filters,
                kernel_size, dilatations, nb_stacks, max_len,
                activation='wavenet', use_skip_connections=True,
                return_param_str=False, output_slice_index=None,
                regression=False):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """
    input_layer = Input(name='input_layer', shape=(max_len, num_feat))
    x = input_layer
    x = Convolution1D(64, kernel_size, padding='same', name='initial_conv')(x)
    x = Convolution1D(nb_filters, kernel_size, padding='same', name='initial_conv_2')(x)
    

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
        x = Dense(num_classes)(x)
        
        x = Activation('softmax', name='output_softmax')(x)
        
        output_layer = x
        print(f'model.x = {input_layer.shape}')
        print(f'model.y = {output_layer.shape}')
        model = Model(input_layer, output_layer)

        adam = optimizers.Adam(lr=0.002, clipnorm=1.)
        model.compile(adam, loss=pinball_loss, metrics=['accuracy'])
        print('Adam with norm clipping.')
    else:
        # regression
        print('1.x.shape=', x.shape)
        x = Dense(128,activation='relu')(x)
        x = Dense(9)(x)
        print('2.x.shape=', x.shape)
        output_layer = Activation('linear', name='output_dense')(x)
        print(f'model.x = {input_layer.shape}')
        print(f'model.y = {output_layer.shape}')
        model = Model(input_layer, output_layer)
        
        
        
        adam = optimizers.Adam(lr=0.01, clipnorm=1.,decay=0.1,amsgrad=True)
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
batch_size = 240
regularization_rate = 0.0001

data_size = 5000
train_data = int(data_size*0.6)
val_data = int(data_size*0.2)


n_input = 5
#用前10个数据预测下一个,第batch_size个数据，n_step个为一组，一个n_input个特征
n_steps = 240
n_hidden = 400
n_class = 1 
n_layers = 18
num_epochs=5000
data_path = "../guoheng/GEFC2014gy.csv"    
period_step = 24
period_day = 7

def dofile(filename,datasize):
    df = pd.read_csv(filename)
    X = [] ; Y = []
    for i in range(datasize-n_steps):
        x = df.loc[i:i+n_steps-1,['load']].values.tolist()
        y = df.loc[i+n_steps,['load']].tolist()
        X.append(x)
        yy = y*9
        Y.append(yy)
    return X,Y

#载入训练数据
x_raw,y_raw = dofile(data_path, data_size)

xtrain = x_raw[0:train_data]
ytrain = y_raw[0:train_data]
x_train=np.array(xtrain)
y_train=np.array(ytrain).reshape(-1,9)

xval = x_raw[train_data:train_data+val_data]
yval = y_raw[train_data:train_data+val_data]
x_val = np.array(xval)
y_val = np.array(yval).reshape(-1,9)

xtest = x_raw[train_data+val_data:data_size-n_steps]
ytest = y_raw[train_data+val_data:data_size-n_steps]
x_test=np.array(xtest)
y_test=np.array(ytest).reshape(-1,9)
    

#x_train, y_train = data_generator(n=200000, seq_length=600)
#x_test, y_test = data_generator(n=40000, seq_length=600)



class PrintSomeValues(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.mape_flag = 100.0
        self.pinball_flag=0.01
    
    
    def on_epoch_begin(self, epoch, logs={}):
        #print(f'x_test[0:1] = {x_test[0:1]}.')
        #print(f'y_test[0:1] = {y_test[0:1]}.')
        #print(f'pred = {self.model.predict(x_test[0:1])}.')
        lr = K.get_value(model.optimizer.lr)
        print("current learning rate is {}".format(lr))
        pred = model.predict(x_test)
        '''
        print('pred.shape:')
        print(pred.shape)
        print('y_test.shape:')
        print(y_test.shape)
        '''
        predict_all = pred.flatten()
        truth_all = y_test.flatten()

        
        mape = eleceval.calcMAPE(predict_all,truth_all)
        mae = eleceval.calcMAE(predict_all,truth_all) 
        mse = eleceval.calcMSE(predict_all,truth_all)
        rmse = eleceval.calcRMSE(predict_all,truth_all) 
        r_2 = eleceval.r2(predict_all,truth_all)
        pinball=pinball_score(y_test,pred,9)
        print("After %d training step(s),"
              "on test data MAPE = %.4f,MAE = %.4f,MSE = %.4f,RMSE = %.4f,R2 = %.4f"\
              % (epoch, mape,mae,mse,rmse,r_2))
        
        if pinball <= self.pinball_flag:
            self.pinball_flag = pinball
            #两个ndarray列合并
            #y_con = np.concatenate((truth_all, predict_all), axis=1)
            a=pinball_score(y_test,pred,9)
            truth_all_reshape=np.reshape(y_test[:,0],[-1,1])
            predict_all_reshape=pred
            y_con = np.concatenate((truth_all_reshape, predict_all_reshape), axis=1)
            #输出真实值和预测值0
            y_out = pd.DataFrame(y_con)
            y_out.to_csv('../guoheng/1h-steps=%d-MAPE=%.4f,MAE = %.4f,MSE = %.4f,RMSE = %.4f,R2 = %.4f,piball_loss = %.4f.csv'% (epoch,mape,mae,mse,rmse,r_2,a))



#def run_task():

model, param_str = dilated_tcn(output_slice_index='last',
                               num_feat=x_train.shape[2],
                               num_classes=0,
                               nb_filters=24,
                               kernel_size=1,
                               dilatations=[0, 1, 2, 3],
                               nb_stacks=8,
                               max_len=x_train.shape[1],
                               activation='norm_relu',
                               use_skip_connections=False,
                               return_param_str=True,
                               regression=True)
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
          batch_size=128,
          initial_epoch=0,
          callbacks=[early_stopping,reduce_lr, psv]
         )

#if __name__ == '__main__':
#    run_task()
