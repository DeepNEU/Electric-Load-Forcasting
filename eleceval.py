
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
from keras.metrics import  mean_absolute_percentage_error
from sklearn.metrics import r2_score
# ¼ÆËãRMSE
def calcRMSE(pred,true):
    return np.sqrt(mean_squared_error(true,pred))

def calcMSE(pred,true):
    return mean_squared_error(true,pred)

# ¼ÆËãMAE
def calcMAE(pred,true):
    #pred = pred[:, 0]
    return mean_absolute_error(true,pred)

# ¼ÆËãMAPE
def calcMAPE(pred,true, epsion = 0.0000000):
    #pred = pred[:,0] # ÁÐ×ªÐÐ£¬±ãÓÚ¹ã²¥¼ÆËãÎó²îÖ¸±ê
    # print (true-pred).shape
    # print true.shape
    # print pred.shape
    #true += epsion
    return np.sum(np.abs((true-pred)/true))*100/len(true)
    #return mean_absolute_percentage_error(true, pred)  返回平均百分比误差

# ¼ÆËãSMAPE
def calcSMAPE(pred,true):
    delim = (np.abs(true)+np.abs(pred))/2.0
    return np.sum(np.abs((true-pred)/delim))/len(true)*100
	


def mape(predicted, test):

    temps = 0.0
    instances = 0
    for i in range(len(predicted)):
        temps += abs(predicted[i]-test[i])*100.0/test[i]
        instances += 1

    return temps/instances

def mpe(predicted, test):
    if not len(predicted) == len(test):
        print("Predicted values and output test instances do not match.")

    temps = 0.0
    instances = 0
    for i in range(len(predicted)):
        temps += (predicted[i]-test[i])*100.0/test[i]
        instances += 1

    return temps/instances

def mse(predicted, test):
    if not len(predicted) == len(test):
        print("Predicted values and output test instances do not match.")

    temps = 0.0
    instances = 0
    for i in range(len(predicted)):
        temps += (predicted[i]-test[i])**2
        instances += 1

    return temps/instances

def rmse(predicted, test):
    return (mse(predicted, test))**0.5

def mae(predicted, test):
    if not len(predicted) != len(test):
        print("Predicted values and output test instances do not match.")

    temps = 0.0
    instances = 0
    for i in range(len(predicted)):
        temps += abs(predicted[i]-test[i])
        instances += 1

    return temps/instances

def r2(predicted, test):
    if not len(predicted) == len(test):
        print("Predicted values and output test instances do not match.")

    return r2_score(test, predicted)

