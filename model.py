# from turtle import dot
import numpy as np
from qp import solve_QP
import quadprog


def linear_kernel(xi, xj):
    """
    Kernel Function, linear kernel (ie: regular dot product)

    :param xi: an input sample (1D np array)
    :param xj: an input sample (1D np array)
    :return: float64
    """
    #TODO
    return np.dot(np.transpose(xi), xj)


def rbf_kernel(xi, xj, gamma=0.1):
    """
    Kernel Function, radial basis function kernel

    :param xi: an input sample (1D np array)
    :param xj: an input sample (1D np array)
    :param gamma: parameter of the RBF kernel (scalar)
    :return: float64
    """
    # TODO
    # X_norm = np.einsum('ij,ij->i',xi,xi)
    # Y_norm = np.einsum('ij,ij->i',xj,xj)
    # K = np.exp(-gamma * (X_norm[:,None] + Y_norm[None,:] - 2 * np.dot(xi, xj)))

    return np.exp(-gamma * np.sum(np.square(xi-xj)))
    #if any buys -> divide by 2 at the end 


def polynomial_kernel(xi, xj, c=2, d=2):
    """
    Kernel Function, polynomial kernel

    :param xi: an input sample (1D np array)
    :param xj: an input sample (1D np array)
    :param c: mean of the polynomial kernel (scalar)
    :param d: exponent of the polynomial (scalar)
    :return: float64
    """
    #TODO
    # inner = (np.transpose(xi)*xj) + c
    inner = np.dot(np.transpose(xi), xj) + c
    return inner**d


class SVM(object):

    def __init__(self, kernel_func=linear_kernel, lambda_param=.1):
        self.kernel_func = kernel_func
        self.lambda_param = lambda_param

    def train(self, inputs, labels):
        """
        train the model with the input data (inputs and labels),
        find the coefficients and constaints for the quadratic program and
        calculate the alphas

        :param inputs: inputs of data, a numpy array
        :param labels: labels of data, a numpy array
        :return: None
        """
        self.train_inputs = inputs
        self.train_labels = labels

        # constructing QP variables
        G = self._get_gram_matrix()
        Q, c = self._objective_function(G)
        A, b = self._inequality_constraint(G)

        # TODO: Uncomment the next line when you have implemented _get_gram_matrix(),
        # _inequality_constraints() and _objective_function().
        self.alpha = solve_QP(Q, c, A, b)[:self.train_inputs.shape[0]]

    def _get_gram_matrix(self):
        """
        Generate the Gram matrix for the training data stored in self.train_inputs.

        Recall that element i, j of the matrix is K(x_i, x_j), where K is the
        kernel function.

        :return: the Gram matrix for the training data, a numpy array
        """
        # TODO 
        #apply kernel to each value inside matrix -> double for loop
        # base_arr = np.zeros([len(self.train_inputs)])
        # for i in range(base_arr[][0]):
        n = len(self.train_inputs)
        gram = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                gram[i, j] = self.kernel_func(self.train_inputs[i], self.train_inputs[j])
        indices = np.triu_indices(n)
        gram[indices] = gram.T[indices]
        return gram

        

    def _objective_function(self, G):
        """
        Generate the coefficients on the variables in the objective function for the
        SVM quadratic program.

        Recall the objective function is:
        minimize (1/2)x^T Q x + c^T x

        :param G: the Gram matrix for the training data, a numpy array
        :return: two numpy arrays, Q and c which fully specify the objective function
        """

        # TODO
        #matrix size 2m x 2m only top left corner is (2 labmba x G) and everything else is zero
        m = self.train_inputs.shape[0]
        Q = np.zeros((2*m, 2*m))
        c = np.zeros(2*m) #length x

        for i in range(m):
            for j in range(m):
                Q[i,j] = (G[i,j] * 2 * self.lambda_param)
        # done finding Q       
        # loop through and set each values
        
        for i in range(m, len(c)): #only loop through second half of the list!
            c[i] = (1/m)
        
        return Q,c



        

    def _inequality_constraint(self, G):
        """
        Generate the inequality constraints for the SVM quadratic program. The
        constraints will be enforced so that Ax <= b.

        :param G: the Gram matrix for the training data, a numpy array
        :return: two numpy arrays, A and b which fully specify the constraints
        """

        # TODO (hint: you can think of x as the concatenation of all the alphas and
        # all the all the xi's; think about what this implies for what A should look like.)
        #Ax \leq B
        #x is a 2m 1D array
        #partition into 4 parts
        #epsilon i geq 0
        m = self.train_inputs.shape[0]
        A = np.zeros((2*m, 2*m))
        b = np.zeros(2*m)
       
        for i in range(m):
            for j in range(m):
                A[i,j] = -1 * self.train_labels[i] * G[i,j] #top left quadrant
        
        for i in range(0,2*m):
            for j in range(m,2*m):
               if (i == j or i + m == j):
                   A[i,j] = -1
        
        for i in range(0,m):
            b[i] = -1

        return A,b 

    def predict(self, inputs):
        """
        Generate predictions given input.

        :param input: 2D Numpy array. Each row is a vector for which we output a prediction.
        :return: A 1D numpy array of predictions.
        """

        #TODO
        #double for loop
        predictions = np.zeros(len(inputs))


        for i in range(len(inputs)):
            count = 0
            for j in range(len(self.train_inputs)):
                count+=(self.alpha[j]*self.kernel_func(self.train_inputs[j],inputs[i]))
                if (count > 0):
                    predictions[i] = 1
                else:
                    predictions[i] = -1
        return predictions


    def accuracy(self, inputs, labels):
        """
        Calculate the accuracy of the classifer given inputs and their true labels.

        :param inputs: 2D Numpy array which we are testing calculating the accuracy of.
        :param labels: 1D Numpy array with the inputs corresponding true labels.
        :return: A float indicating the accuracy (between 0.0 and 1.0)
        """

        #TODO
        predictions = self.predict(inputs)
        return np.mean(predictions == labels) #ratio of number of correct matches
