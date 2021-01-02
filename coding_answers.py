import numpy as np
import matplotlib.pyplot as plt

############################# Functions ############################

#build design matrix for polynomial basis functions of order order 
def build_phi_polynomial(order,X):
    phi = np.zeros([len(X),order+1])
    for data_point_idx in range(len(X)):
        for basis_idx in range(order+1):
            phi[data_point_idx, basis_idx] = X[data_point_idx]**basis_idx
    return phi 


#build design matrix for trigonometric basis functions of order order
def build_phi_trig(order, X):
    phi = np.zeros([len(X),2*order+1])
    for data_point_idx in range(len(X)):
        for j in range(order+1):
            phi[data_point_idx, 2 * j - 1] = np.sin(2 * np.pi * j * X[data_point_idx])
            phi[data_point_idx, 2 * j] = np.cos(2 * np.pi * j * X[data_point_idx])
    return phi 


#obtain w* according to derived max likelihood solution in report
def get_w_hat(phi,Y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi),phi)), np.transpose(phi)), Y)


#build polynomial function for plotting outside training data range
def build_polynomial(x, w_hat, order):
    polynomial_array = np.array([x**i for i in range(order+1)])
    return np.dot(w_hat[:,0],polynomial_array)


#build trigonometric function for plotting outside training data range
def build_trig_function(x, w_hat, order):
        #if w_hat.shape[1] > 0
        trig_array = np.zeros(2*order+1)
        #set first basis function to 1
        trig_array[0] = 1
        for j in range(1,order+1):
            trig_array[2*j-1] = np.sin(2 * np.pi * j * x)
            trig_array[2*j] = np.cos(2 * np.pi * j * x)
        
        return np.dot(w_hat[:,0],trig_array)
    
    
# helper function to plot predicted mean in the case of polynomial basis functions in interval [-0.3,1.3]
def plot_predicted_means(order, X, Y):
    X_pred = np.linspace(-0.3,1.3,100)
    phi = build_phi_polynomial(order,X)
    w_hat = get_w_hat(phi, Y)
    
    #get predicted Y for every X_pred in interval [-0.3,1.3]
    Y_pred = np.array([build_polynomial(X_pred[i], w_hat, order) for i in range(len(X_pred))])
    plt.plot(X_pred,Y_pred, label='Order {}'.format(order),linewidth = 1.5)
    plt.legend()
    
    
#plot all predicted means in case of polynomial basis functions for orders that are given in order_array    
def plot_different_orders(order_array, X, Y):
    fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
    
    for order in order_array:
        plot_predicted_means(order, X, Y)
        
    plt.scatter(X,Y, color='r',label='Original Data', marker='x', s = 150)
    plt.legend()
    plt.ylim(-1.2,1.3)
    plt.title('Linear Regression - Polynomial Basis Functions')
    plt.savefig('exercise_a')
    

# helper function to plot predicted mean in the case of trigonometric basis functions in interval [-0.3,1.3]
def plot_predicted_means_trig(order, X, Y):
    X_pred = np.linspace(-1,1.2,1000)
    phi = build_phi_trig(order,X)
    w_hat = get_w_hat(phi, Y)
    
    Y_pred = np.array([build_trig_function(X_pred[i], w_hat, order) for i in range(len(X_pred))])
    plt.plot(X_pred,Y_pred, label='Order {}'.format(order),linewidth = 1.5)
    plt.legend()    

    
#plot all predicted means in case of trigonometric basis functions for orders that are given in order_array    
def plot_different_orders_trig(order_array, X, Y):
    fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
    
    for order in order_array:
        plot_predicted_means_trig(order, X, Y)
        
    plt.scatter(X,Y, color='r',label='Original Data', marker='x', s = 150)
    plt.legend()
    plt.ylim(-1.2,1.2)
    plt.title('Linear Regression - Trigonometric Basis Functions')
    plt.savefig('exercise_b')
    

#return average square error and average sigma squared for leave-one-out cross validation of predicted means in the case of trigonometric basis function of order order
def leave_one_out_error(order,X,Y):
    #initialize array for squared errors of test point and sigma squareds
    squared_errors = np.zeros(len(X))
    sigmas_squared = np.zeros(len(X))
    #iterate over every data point for leave one out
    for i, value in enumerate(X):
        X_test = value
        Y_test = Y[i]
        X_train =  np.delete(X, i)
        Y_train = np.delete(Y, i)
        #build phi according to new training data
        phi = build_phi_trig(order, X_train)
        #get parameter estimate on Y_train
        w_hat = get_w_hat(phi, Y_train)
        #calculate sigma_squared
        Y_difference = Y_train - np.matmul(phi, w_hat)
        sigmas_squared[i] = 1/len(X_train)*np.sum(Y_difference**2)
        #resize to bring to correct shape
        w_hat = np.resize(w_hat,(w_hat.shape[0],1))
        #predict Y for left out test point
        Y_pred = build_trig_function(X_test, w_hat, order)
        #calculate squared error
        squared_errors[i] = (Y_pred - Y_test)**2
    
    #return average of squared errors
    return np.mean(squared_errors), np.mean(sigmas_squared)


#create plot for Exercise 1c)
def plot_error_and_sigma_squared(order_array, X, Y):
    sigmas_squared = np.zeros(len(order_array))
    avg_squared_errors = np.zeros(len(order_array))
    
    for i, order in enumerate(order_array):
        avg_squared_errors[i], sigmas_squared[i] = leave_one_out_error(order, X, Y)
    
    fig=plt.figure(figsize=(10,10), dpi= 100, facecolor='w', edgecolor='k')
    plt.xlabel('Order of Trigonometric Basis Function')
    plt.plot(order_array, sigmas_squared, label='Average σ^2')
    plt.plot(order_array, avg_squared_errors, label='Average Squared Error')
    plt.legend()
    plt.title('Model Selection through Leave One Out Cross Validation')
    plt.savefig('exercise_c')
####################################################################    
    
    
############################# Execution ############################

if __name__ == '__main__':
    #build data
    N = 25
    X = np.reshape(np.linspace(0, 0.9, N), (N, 1)) 
    Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)
    
    #Plot for exercise 1a)
    order_array=np.array([0,1,2,3,11])
    plot_different_orders(order_array, X, Y)
    print('Saved plot of predicted means of polynomial basis functions of orders {} in current folder'.format(order_array))
    
    #Plot for exercise 1b)
    order_array=np.array([1,11])
    plot_different_orders_trig(order_array, X, Y)
    print('Saved plot of predicted means of trigonometric basis functions of orders {} in current folder'.format(order_array))
    
    
    #Plot for exercise 1c)
    order_array = [0,1,2,3,4,5,6,7,8,9,10]
    plot_error_and_sigma_squared(order_array, X, Y)
    print('Saved plot of Average Squared Error and Averaged σ^2 for orders from {} to {} inclusive in current folder'.format(order_array[0], order_array[-1]))
    

    