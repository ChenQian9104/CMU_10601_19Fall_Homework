#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:25:19 2019

@author: qianchen
"""

import csv
import numpy as np 
class Linear_layer:    
    def __init__(self, weights):
        self.weights = weights
    
    def forward(self, x):
        return np.dot(self.weights, x)
    
    def backward(self, x, a, ga):
        
        g_alpha = np.dot( ga, x.T)
        gx = np.dot(self.weights.T, ga)
        return g_alpha, gx        

class Sigmoid_layer:
    
    def forward(self, a):
        return 1/( 1 + np.exp(-a) )
    
    def backward(self, a, z, gz):
        ga = (1-z)*z*gz
        return ga
    
    
class CrossEntropy_Softmax_layer:
    
    
    def forward(self, b, label):
        
        K = b.shape[0]
        N = b.shape[1]
        y_hat = np.zeros( (K, N) )
        
        
        # softmax 
        for i in range(N):
            exp_sum = np.sum( np.exp( b[:, i] ) )
            
            y_hat[:,i] = np.exp( b[:,i] )/exp_sum
        
        # loss function
        loss = 0.0
        for i in range(N):
            loss += -np.sum( np.multiply( label[:,i], np.log(y_hat[:,i])  ) )
        loss /= N
        
        return y_hat, loss 
    
    def backward(self, y_hat, y):
        
        dJdy = self.backward_cross_entropy(y_hat,y)
        dydb = self.backward_softmax(y_hat)
        
        gb = np.dot( dydb, dJdy)
        
        return gb
    
    def backward_cross_entropy(self,y_hat, y):
        
        K = y_hat.shape[0]
        
        dJdy = np.zeros( (K,1) )
        
        dJdy = -np.multiply( y, 1/y_hat)
        
        
        
        
        return dJdy    # dJ/dy
    
        
        
    def backward_softmax(self, y_hat):
        
        
        dydb = np.diag( y_hat.ravel())  - np.dot( y_hat, y_hat.T)
        
        return dydb
        
        
            
class NN:
    def __init__(self, input_units, hidden_units, output_units, lr, init_flag):
        
        if init_flag == 1:
            self.W1 = self.weights_initialize_uniform(  hidden_units,input_units + 1)
            self.W2 = self.weights_initialize_uniform(  output_units, hidden_units + 1 )
        
        if init_flag == 2:
            self.W1 = self.weights_initialize_zero( hidden_units,input_units + 1 )
            self.W2 = self.weights_initialize_zero( output_units, hidden_units + 1)
            
        self.input_layer = Linear_layer(self.W1)
        self.hidden_layer = Sigmoid_layer()
        self.output_layer = Linear_layer(self.W2)
        self.loss_function = CrossEntropy_Softmax_layer()
        
        self.lr = lr # learning rate
        
        
       
        
    def weights_initialize_uniform( self, d1, d2 ):
        W = np.random.uniform(-0.1, 0.1, size =(d1,d2) )
        W[:,0] = 0.0
        return W
        
    def weights_initialize_zero(self,d1, d2):
        W = np.zeros( (d1,d2) )
        return W        
        
    def forward_computation(self, data, label): 
        a = self.input_layer.forward(data)
        z = self.hidden_layer.forward(a)
        
        # Add bias term to z 
        N = data.shape[1]
        bias_term = np.ones( (1, N))
        z = np.vstack( (bias_term,z) )

        b = self.output_layer.forward(z)
        y_hat, loss = self.loss_function.forward(b,label)
       
        
        return a,z,b,y_hat, loss
        
    def backward_computation(self, data, label, a,z,b,y_hat):
        
 
        gb = self.loss_function.backward(y_hat,label)
        
        g_beta, gz = self.output_layer.backward(z,b,gb)
        
        ga = self.hidden_layer.backward(a[1:],z[1:],gz[1:,])
        
        g_alpha, gx = self.input_layer.backward(data,a,ga)
        
        
        return g_alpha, g_beta
    
    def update_weights(self, g_alpha,g_beta):
        
        self.W1 -= self.lr*g_alpha
        self.W2 -= self.lr*g_beta 
        
    def train(self, data, label, num_epoch):
        
        for _ in range(num_epoch):
            
            N = data.shape[1]
            
            for i in range(N):
                
                # forward computation
                a, z, b, y_hat, loss = self.forward_computation(data, label)
                g_alpha, g_beta = self.backward_computation(data[:,i][:,None], label[:,i][:,None], a[:,i][:,None],
                                                            z[:,i][:,None], b[:,i][:,None], y_hat[:,i][:,None])
                #if i%10 == 0: print(loss)
                
                self.update_weights(g_alpha, g_beta)
                
    def predict(self, data):
        a = self.input_layer.forward(data)
        z = self.hidden_layer.forward(a)
        
        # Add bias term to z 
        N = data.shape[1]
        bias_term = np.ones( (1, N))
        z = np.vstack( (bias_term,z) )

        b = self.output_layer.forward(z)  
        
        K = b.shape[0]
        N = b.shape[1]
        y_hat = np.zeros( (K, N) )
        pred_label = np.zeros( (N,1) )
        # softmax 
        for i in range(N):
            exp_sum = np.sum( np.exp( b[:, i] ) )
            
            y_hat[:,i] = np.exp( b[:,i] )/exp_sum   
            
            pred_label[i] = np.argmax( y_hat[:,i] )
        return y_hat , pred_label
        
def load_data(file_path):
    
    with open(file_path) as f:
        reader = csv.reader(f)
        content = np.array( list(reader) )
        label_train = content[:,0].astype(int)
        data_train = content[:,1:].astype(int)    
        
        
        
        N = label_train.shape[0] # number of samples 
        K = 10 # class number 
        M = data_train.shape[1] # number of features
        
        label = np.zeros(  (K,N)  ) 
        data = np.zeros( (M+1, N) )
        
        data[0,:] = 1.0   # add bias term
        
        for i in range(N):
            data[1:, i] = (data_train[i,:]).T
            
            label[  label_train[i], i] = 1.0
            
        return data, label, label_train
    


if __name__ == '__main__':
    
    train_input = "smalltrain.csv"
    valid_input = "smallvalidation.csv"
    train_out  = "model1train_out.labels"
    test_out = "model1test_out.labels"
    metrics_out = "model1metrics_out.txt"
    num_epoch = 2
    hidden_units = 4
    init_flag = 2
    lr = 0.1
    
    data_train, label_train, label_train_1d = load_data(train_input)
    data_test,  label_test, label_test_1d  = load_data(valid_input)
    input_units = data_train.shape[0] -1 
    output_units = label_train.shape[0]
    
    model = NN(input_units, hidden_units, output_units, lr, init_flag)
    
    cross_entropy_train = []
    cross_entropy_test = []
    for i in range(num_epoch):
        model.train(data_train, label_train, 1)
        _, _, _, _, loss_train = model.forward_computation(data_train, label_train)
        _, _, _, _, loss_test  = model.forward_computation(data_test,  label_test)
        print("epoch = " + str(i) +' ' + str(loss_train) )
        print("epoch = " + str(i) +' ' + str(loss_test) )
        
        cross_entropy_train.append(loss_train)
        cross_entropy_test.append(loss_test)
    
    
    _, pred_label_train = model.predict(data_train)
    _, pred_label_test = model.predict(data_test)
    
    error_train, error_test = 0,0 
    
    error_count = 0.0
    num_train = pred_label_train.shape[0]
    for i in range( num_train ):
        if pred_label_train[i] == label_train_1d[i]:
            error_count += 1
    error_train = 1  - error_count/num_train
    print("error rate is:", error_train)
    
    error_count = 0.0
    num_test = pred_label_test.shape[0]
    for i in range( num_test ):
        if pred_label_test[i] == label_test_1d[i]:
            error_count += 1
    error_test = 1 - error_count/num_test
    print("error rate is:", error_test)    
    
    with open(train_out,'w') as f:
        for label in pred_label_train:
            f.write(str(int(label) ) + '\n')
            
    with open(test_out,'w') as f:
        for label in pred_label_test:
            f.write(str(int(label)) + '\n' )
            
    with open(metrics_out,'w') as f:
        for i in range(num_epoch):
            f.write("epoch=" + str(i+1) + " ")
            f.write("crossentropy(train): " + str( cross_entropy_train[i] ) + '\n')
            
            f.write("epoch=" + str(i+1) + " ")
            f.write("crossentropy(test): " + str( cross_entropy_test[i] ) + '\n') 
            
        f.write("error(train): " + str(error_train) + "\n")
        f.write("error(test): " + str(error_test) )
            

        
        
    
    
    
            
        
        
        
        
            
        
        
    

        
    
            
        
        
        