#Author @ Jack Volonte
#Date 9/16/2022
#Description : This file performs a cost optimization algorithim implementing a batch gradiant descent algorithim, displaying the different regression lines
# and the error loss function for each batch size over time - for the dipnet fishing harvest data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


# read in the data and assign to variables
data = pd.read_csv('dip-har-eff.csv')
data = data.values
x = data[:, 1]
y = data[:, 2]

# normalize the data
x_keep = x
y_keep = y
x = x / (max(x) - min(x))
y = y / (max(y) - min(y))

# parameter values
alpha = .1 # learning rate
m = y.size # size of x & y array

# gradient descent function
def gradient_descent(x, y, alpha, batch):
    # initalizing theta values
    theta_0 = np.array([0.0])   # theta_0 starting guess
    theta_1 = np.array([0.0])   # theta_1 starting guess

    # initalize epoch
    epoch = 5000

    # inital mse
    mse = sum([(theta_0 + theta_1 * x[i] - y[i]) ** 2 for i in range(m)])

    #print((x[i] for i in range(batch)))

    # for loop, iterates for time of epoch
    for i in range(epoch):
        grad_0 = (1/m) * sum([(theta_0 + theta_1 * x[i] - y[i]) for i in range(batch)])
        grad_1 = (1/m) * sum([(theta_0 + theta_1 * x[i] - y[i]) * x[i] for i in range(batch)])
        temp_0 = theta_0 - alpha * grad_0
        temp_1 = theta_1 - alpha * grad_1
        theta_0 = temp_0
        theta_1 = temp_1

        new_mse = sum([(theta_0 + theta_1 * x[i] - y[i]) ** 2 for i in range(m)])
        mse = new_mse

    # cost optimization plot names
    plt.xlabel('days fishes / permits issued')
    plt.ylabel('harvest - salmon caught')
    plt.title('days fished vs total harvest')
    plt.scatter(x, y)
    plt.plot([min(x), max(x)], theta_1 * [min(x), max(x)] + theta_0)  # line for each batch

    plt.plot([min(x), max(x)], [min(y), max(y)], color='blue')  # regression line

    #print("slope : ", theta_1)
    #print("intercept: ", theta_0)
    #print("error: ", mse)


# plot error function, plots error for different batch sizes
def ploterror(x,y,alpha,batch):
    # initalizing theta values
    theta_0 = np.array([0.0])  # theta_0 guess
    theta_1 = np.array([0.0])  # theta_1 guess

    # initalize epoch
    epoch = 500

    # for loop, iterates for time of epoch
    for i in range(epoch):
        grad_0 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) for i in range(batch)])
        grad_1 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) * x[i] for i in range(batch)])
        temp_0 = theta_0 - alpha * grad_0
        temp_1 = theta_1 - alpha * grad_1
        theta_0 = temp_0
        theta_1 = temp_1


    theta_value = np.linspace(0, epoch, m)

    sqrerror_out = [(theta_1*x[i] - y[i]) ** 2 for i in range(m)]
    plt.plot(theta_value, sqrerror_out)

# find R^2 valued of each batch size
def findR2(x,y,alpha,batch):
    # initalizing theta values
    theta_0 = np.array([0.0])  # theta_0 guess
    theta_1 = np.array([0.0])  # theta_1 guess

    # initalize epoch
    epoch = 500

    # for loop, iterates for time of epoch
    for i in range(epoch):
        grad_0 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) for i in range(batch)])
        grad_1 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) * x[i] for i in range(batch)])
        temp_0 = theta_0 - alpha * grad_0
        temp_1 = theta_1 - alpha * grad_1
        theta_0 = temp_0
        theta_1 = temp_1

    y_pred = [y[i] - (theta_1 * x[i]) ** 2 for i in range(m)]
    r = r2_score(y, y_pred)
    return r

def y_pred(x,y,alpha,batch):
    # initalizing theta values
    theta_0 = np.array([0.0])  # theta_0 guess
    theta_1 = np.array([0.0])  # theta_1 guess

    # initalize epoch
    epoch = 500

    # for loop, iterates for time of epoch
    for i in range(epoch):
        grad_0 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) for i in range(batch)])
        grad_1 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) * x[i] for i in range(batch)])
        temp_0 = theta_0 - alpha * grad_0
        temp_1 = theta_1 - alpha * grad_1
        theta_0 = temp_0
        theta_1 = temp_1

    y_pred = [y[i] - (theta_1 * x[i]) ** 2 for i in range(m)]
    return y_pred

# rmse for each batch size
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def rsmeplots(x,y,alpha,batch):
    # initalizing theta values
    theta_0 = np.array([0.0])  # theta_0 guess
    theta_1 = np.array([0.0])  # theta_1 guess

    # initalize epoch
    epoch = 500

    # for loop, iterates for time of epoch
    for i in range(epoch):
        grad_0 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) for i in range(batch)])
        grad_1 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) * x[i] for i in range(batch)])
        temp_0 = theta_0 - alpha * grad_0
        temp_1 = theta_1 - alpha * grad_1
        theta_0 = temp_0
        theta_1 = temp_1


    theta_value = np.linspace(0, epoch, m)

    sqrerror_out = y_pred(x,y,alpha,batch)
    plt.plot(theta_value, sqrerror_out)

def thetaReturns(x,y,alpha,batch):
    # initalizing theta values
    theta_0 = np.array([0.0])  # theta_0 guess
    theta_1 = np.array([0.0])  # theta_1 guess

    # initalize epoch
    epoch = 500

    # for loop, iterates for time of epoch
    for i in range(epoch):
        grad_0 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) for i in range(batch)])
        grad_1 = (1 / m) * sum([(theta_0 + theta_1 * x[i] - y[i]) * x[i] for i in range(batch)])
        temp_0 = theta_0 - alpha * grad_0
        temp_1 = theta_1 - alpha * grad_1
        theta_0 = temp_0
        theta_1 = temp_1

    y_pred = [y[i] - (theta_1 * x[i]) ** 2 for i in range(m)]
    return (theta_1,theta_0)

# 2023 predictions
def predAhead(theta1,theta0,xToTest):
    return (theta1 * xToTest + theta0) * (max(x) - min(x))




#batch size 1
gradient_descent(x,y,alpha,1)
plt.show()

#batch size 5
gradient_descent(x,y,alpha,5)
plt.show()

#batch size 10
gradient_descent(x,y,alpha,10)
plt.show()

#batch size 19
gradient_descent(x,y,alpha,19)
plt.show()

# all batch sizes plots on one plot
gradient_descent(x,y,alpha,1)
gradient_descent(x,y,alpha,5)
gradient_descent(x,y,alpha,10)
gradient_descent(x,y,alpha,19)
plt.show()

#error loss function plots
plt.xlabel('training period - 500 epochs')
plt.ylabel('error loss')
plt.title('error loss over time - different batch sizes')
ploterror(x,y,alpha,1)
ploterror(x,y,alpha,5)
ploterror(x,y,alpha,10)
ploterror(x,y,alpha,19)
plt.show()

# plot normal x y graph w/ regression line
plt.xlabel('days fishes / permits issued')
plt.ylabel('harvest - salmon caught')
plt.title('days fished vs total harvest')
plt.xlim(0,x.max())
plt.ylim(0,y.max())
plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(y), max(y)], color='red')  # regression line
plt.show()

# r squared for each batch size
r = findR2(x,y,alpha,1)
print("R^2 for batch of 1:", r)

r = findR2(x,y,alpha,5)
print("R^2 for batch of 5:", r)

r = findR2(x,y,alpha,10)
print("R^2 for batch of 10:", r)

r = findR2(x,y,alpha,19)
print("R^2 for batch of 19:", r)

# rsme plots for different batch sizes
plt.xlabel('training period - 500 epochs')
plt.ylabel('RSME')
plt.title('RSME over time - batch size of 1')
rsmeplots(x,y,alpha,1)
plt.show()
plt.xlabel('training period - 500 epochs')
plt.ylabel('RSME')
plt.title('RSME over time - batch size of 5')
rsmeplots(x,y,alpha,5)
plt.show()
plt.xlabel('training period - 500 epochs')
plt.ylabel('RSME')
plt.title('RSME over time - batch size of 10')
rsmeplots(x,y,alpha,10)
plt.show()
plt.xlabel('training period - 500 epochs')
plt.ylabel('RSME')
plt.title('RSME over time - batch size of 19')
rsmeplots(x,y,alpha,19)
plt.show()

# rmse for each batch size
print("RMSE for batch size 1 = ",rmse(y,y_pred(x,y,alpha,1)))
print("RMSE for batch size 5 = ",rmse(y,y_pred(x,y,alpha,5)))
print("RMSE for batch size 10 = ",rmse(y,y_pred(x,y,alpha,10)))
print("RMSE for batch size 19 = ",rmse(y,y_pred(x,y,alpha,19)))

# prediction for 2023
theta1,theta0 = thetaReturns(x,y,alpha,19)
print("2023 prediction with 28,375 days of fishing = ", predAhead(theta1,theta0,(28375/(max(x_keep)-min(x_keep))))*(max(y_keep)-min(y_keep)))