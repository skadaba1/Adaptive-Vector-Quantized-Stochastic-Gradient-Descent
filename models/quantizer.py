
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle
from multiprocessing import Pool
import random
import itertools
import time
import math
import scipy
from multiprocessing import Process

class SimplexQuantizer():
    def __init__(self, iters, dimC):
        self.iters = iters;
        self.C = None;
        self.EA = None;
        self.Eb = None;
        self.dimC = dimC;
        
    def load(self, grads):
        self.grads = grads;
        self.d = np.shape(grads)[0];
        self.m = np.shape(grads)[1];
        self.betas = np.ones((self.m,));
        self.points = []
        for i in range(self.dimC):
            xn = np.random.normal(0, 1, self.d);
            xn = xn * 1/np.linalg.norm(xn);
            self.points.append(xn);

    def objective(self, C, A):
        #print(np.shape(C), np.shape(A), np.shape(self.grads))
        total = 0;
        for i in range(self.m):
            total += self.betas[i]*np.square(np.linalg.norm(self.grads[:,i] - C@A))/np.square(np.linalg.norm(self.grads[i]))
        return total/self.m
    
    def compute(self, verbose):
        # Compute minimum covering ellipsoid #
        A = cp.Variable((self.d, self.d), PSD=True);
        b = cp.Variable(self.d);
        objective = cp.Minimize(-cp.log_det(A));
        constraints = [A == A.T];
        for i in range(self.m):
            constraints.append(cp.norm(A@self.grads[:,i] + b) <= 1);
        prob = cp.Problem(objective, constraints);
        print("Starting to compute Lowner-John Ellipsoid...")
        opt = prob.solve('SCS', eps=1e-2, max_iters = 50);
        A = A.value if A.value is not None else self.EA;
        b = b.value if b.value is not None else self.Eb;
        w, v = np.linalg.eigh(A);
        print("First solve status = ", prob.status)
        
        self.EA = A;
        self.Eb = b
        prob = None;
        indexes = range(self.dimC)
        pairs = list(itertools.combinations(indexes, 2))
        eijs = [np.zeros((self.dimC)) for i in range(len(pairs))];
        for index, pair in enumerate(pairs):
            eijs[index][pair[0]] = 1;
            eijs[index][pair[1]] = -1;
        
        # compute set C using N discrete points with SDP relaxation#
        R = np.linalg.norm(np.sum(self.grads, 1))
        X = cp.Variable((self.d, self.dimC));
        Y = cp.Variable((self.dimC, self.dimC));
        alpha = cp.Variable(1);
        B = cp.vstack([cp.hstack([np.identity(self.d), X]), cp.hstack([X.T, Y])])
        upper_bound = w[0];
        lower_bound = w[1]/self.dimC;
        constraints = [B >= 0, cp.diff(X, k=1, axis=1) <= R];
        expr = alpha + cp.sum(cp.diff(X, k=1, axis=1));
        for i in range(self.dimC):
            if(i < self.grads.shape[1]):
                expr += cp.norm(X[:,i] - self.grads[:, -i]);
            else:
                expr += cp.norm(X[:,i] - np.linalg.inv(A)@(random.choice(self.points) - b))
            constraints.append(cp.norm(A@X[:,i] + b) <= 1)
        for i in range(len(eijs)):
            constraints.append(- alpha <= eijs[i].T@Y@eijs[i]);
            constraints.append(alpha >= eijs[i].T@Y@eijs[i]);
        objective = cp.Minimize(expr);
        prob = cp.Problem(objective, constraints);
        flag = False;
        try:
            opt = prob.solve('ECOS', eps=1e-2)
        except:
            flag = True
        ## Catch infeasible case ##
        if(prob.status not in ['optimal', 'optimal_inaccurate'] or flag):
            print("Defaulted to border fitting");
            X = cp.Variable((self.d, self.dimC));
            alpha = cp.Variable(1);
            expr = 0; constraints = [];
            for i in range(self.dimC):
                if(i < self.grads.shape[1]):
                    expr += cp.norm(X[:,i] - self.grads[:, -i]);
                else:
                    expr += cp.norm(X[:,i] - np.linalg.inv(A)@(random.choice(self.points) - b))
                constraints.append(cp.norm(A@X[:,i] + b) <= 1 + alpha)
            objective = cp.Minimize(expr);
            prob = cp.Problem(objective, constraints);
            opt = prob.solve('ECOS');
            
        print("Second solve status = ", prob.status)
        self.C = self.C if X.value is None else X.value
        
        return opt
    
    def plot(self, grads):
        A = self.EA;
        b = self.Eb
        r = 1;
        #The lower this value the higher quality the circle is with more points generated
        stepSize = 0.01
        #Generated vertices
        positions = []
        t = 0
        while t < 2 * math.pi:
            positions.append(np.array([r * math.cos(t), r * math.sin(t)]))
            t += stepSize
        ellipsoid_coords_x = []
        ellipsoid_coords_y = []
        for pos in positions:
            point = np.linalg.inv(A)@(pos - b)
            ellipsoid_coords_x.append(point[0]);
            ellipsoid_coords_y.append(point[1]);
        plt.figure(1);
        plt.plot(grads[0,:], grads[1,:], 'b.', label='True grads');
        plt.plot(self.C[0,:], self.C[1,:], 'g.', label='Discrete (bin) points');
        plt.plot(ellipsoid_coords_x, ellipsoid_coords_y, 'r-', label='Parameterized ellipse');
        plt.legend(loc='upper right')
    
    def quantize(self, g):
        val = 0;
        index = 0;
        min_dist = float('inf');
        for i in range(self.dimC):
            dist = np.linalg.norm(self.C[:, i] - g)
            if(dist < min_dist):
                min_dist = dist;
                index = i;
        val = min_dist;
        print("Minimum relative distance = ", min_dist/np.linalg.norm(g));
        return (index, val)
    

class EllipsoidQuantizer():
    def __init__(self, iters, dimC):
        self.iters = iters;
        self.C = None;
        self.EA = None;
        self.Eb = None;
        self.dimC = dimC;
        
    def load(self, grads):
        self.grads = grads;
        self.d = np.shape(grads)[0];
        self.m = np.shape(grads)[1];
        self.betas = np.ones((self.m,));
        self.points = []
        for i in range(self.dimC):
            xn = np.random.normal(0, 1, self.d);
            xn = xn * 1/np.linalg.norm(xn);
            self.points.append(xn);

    def objective(self, C, A):
        #print(np.shape(C), np.shape(A), np.shape(self.grads))
        total = 0;
        for i in range(self.m):
            total += self.betas[i]*np.square(np.linalg.norm(self.grads[:,i] - C@A))/np.square(np.linalg.norm(self.grads[i]))
        return total/self.m
    
    def compute(self, verbose):
        # Compute minimum covering ellipsoid #
        A = cp.Variable((self.d, self.d), PSD=True);
        b = cp.Variable(self.d);
        objective = cp.Minimize(-cp.log_det(A));
        constraints = [A == A.T];
        for i in range(self.m):
            constraints.append(cp.norm(A@self.grads[:,i] + b) <= 1);
        prob = cp.Problem(objective, constraints);
        print("Starting to compute Lowner-John Ellipsoid...")
        opt = prob.solve('SCS', eps=1e-2, max_iters = 50);
        A = A.value if A.value is not None else self.EA;
        b = b.value if b.value is not None else self.Eb;
        w, v = np.linalg.eigh(A);
        print("First solve status = ", prob.status)
        
        self.EA = A;
        self.Eb = b
        prob = None;
        indexes = range(self.dimC)
        pairs = list(itertools.combinations(indexes, 2))
        eijs = [np.zeros((self.dimC)) for i in range(len(pairs))];
        for index, pair in enumerate(pairs):
            eijs[index][pair[0]] = 1;
            eijs[index][pair[1]] = -1;
        
        # compute set C using N discrete points with SDP relaxation#
        R = np.linalg.norm(np.sum(self.grads, 1))
        X = cp.Variable((self.d, self.dimC));
        Y = cp.Variable((self.dimC, self.dimC));
        alpha = cp.Variable(1);
        B = cp.vstack([cp.hstack([np.identity(self.d), X]), cp.hstack([X.T, Y])])
        upper_bound = w[0];
        lower_bound = w[1]/self.dimC;
        constraints = [B >= 0, cp.diff(X, k=1, axis=1) <= R];
        expr = alpha + cp.sum(cp.diff(X, k=1, axis=1));
        for i in range(self.dimC):
            if(i < self.grads.shape[1]):
                expr += cp.norm(X[:,i] - self.grads[:, -i]);
            else:
                expr += cp.norm(X[:,i] - np.linalg.inv(A)@(random.choice(self.points) - b))
            constraints.append(cp.norm(A@X[:,i] + b) <= 1)
        for i in range(len(eijs)):
            constraints.append(- alpha <= eijs[i].T@Y@eijs[i]);
            constraints.append(alpha >= eijs[i].T@Y@eijs[i]);
        objective = cp.Minimize(expr);
        prob = cp.Problem(objective, constraints);
        flag = False;
        try:
            opt = prob.solve('ECOS', eps=1e-2)
        except:
            flag = True
        ## Catch infeasible case ##
        if(prob.status not in ['optimal', 'optimal_inaccurate'] or flag):
            print("Defaulted to border fitting");
            X = cp.Variable((self.d, self.dimC));
            alpha = cp.Variable(1);
            expr = 0; constraints = [];
            for i in range(self.dimC):
                if(i < self.grads.shape[1]):
                    expr += cp.norm(X[:,i] - self.grads[:, -i]);
                else:
                    expr += cp.norm(X[:,i] - np.linalg.inv(A)@(random.choice(self.points) - b))
                constraints.append(cp.norm(A@X[:,i] + b) <= 1 + alpha)
            objective = cp.Minimize(expr);
            prob = cp.Problem(objective, constraints);
            opt = prob.solve('ECOS');
            
        print("Second solve status = ", prob.status)
        self.C = self.C if X.value is None else X.value
        
        return opt
    
    def plot(self, grads):
        A = self.EA;
        b = self.Eb
        r = 1;
        #The lower this value the higher quality the circle is with more points generated
        stepSize = 0.01
        #Generated vertices
        positions = []
        t = 0
        while t < 2 * math.pi:
            positions.append(np.array([r * math.cos(t), r * math.sin(t)]))
            t += stepSize
        ellipsoid_coords_x = []
        ellipsoid_coords_y = []
        for pos in positions:
            point = np.linalg.inv(A)@(pos - b)
            ellipsoid_coords_x.append(point[0]);
            ellipsoid_coords_y.append(point[1]);
        plt.figure(1);
        plt.plot(grads[0,:], grads[1,:], 'b.', label='True grads');
        plt.plot(self.C[0,:], self.C[1,:], 'g.', label='Discrete (bin) points');
        plt.plot(ellipsoid_coords_x, ellipsoid_coords_y, 'r-', label='Parameterized ellipse');
        plt.legend(loc='upper right')
    
    def quantize(self, g):
        val = 0;
        index = 0;
        min_dist = float('inf');
        for i in range(self.dimC):
            dist = np.linalg.norm(self.C[:, i] - g)
            if(dist < min_dist):
                min_dist = dist;
                index = i;
        val = min_dist;
        print("Minimum relative distance = ", min_dist/np.linalg.norm(g));
        return (index, val)

## Quantizer NN
class NNQuantizer():
    def __init__(self, iters, dimC):
        self.iters = iters
        self.C = None
        self.A = None
        self.dimC = dimC # How many gradients to keep track of
        
    # What is this function doing
    def load(self, grads):
        self.grads = grads
        self.m = np.shape(grads)[0]
        self.d = (np.shape(grads)[1], np.shape(grads)[2])
        self.betas = np.ones((self.m,)); # Decay vector


    def dot(self, C, A):
        return np.tensordot(A, C, axes=([0],[0]))


    def objective(self, C, A):
        #print(np.shape(C), np.shape(A), np.shape(self.grads))
        total = 0;
        for i in range(self.m):
            # Divinding by norm of gradients for numerical stability
            total += 0.5 * np.square(np.linalg.norm(self.dot(C, A) - self.grads[i])) / np.square(np.linalg.norm(self.grads[i]))
        return total / self.m


    def find(self, u, sv):
        condition = u > ( (sv-1) / range(1,len(u)+1) )
        for i in reversed(range(len(condition))):
            if condition[i]:
                return i


    def projToSmplx(self, v):
        u = np.sort(v)[::-1]
        sv = np.cumsum(u)
        ind = self.find(u,sv)
        theta = (sv[ind]-1) / (ind+1)
        x = np.maximum(v - theta, 0)
        return x
    

    def grad_C(self, C, A):
        g = np.zeros(np.shape(C))
        for i in range(self.m):
            inside = self.dot(C, A) - self.grads[i]
            inside = inside[np.newaxis, :, :]
            add = A[:, np.newaxis, np.newaxis] * np.repeat(inside, C.shape[0], axis = 0)
            # add = np.expand_dims((C@A - self.grads[:,i]),1)@np.expand_dims(A, 1).T
            # add = self.betas[i] * np.outer(C@A - self.grads[:,i], A)
            g += add
        return g


    def grad_A(self, C, A, index):
        g = np.zeros(np.shape(A));
        for i in range(self.m):
            add = np.tensordot(C, self.dot(C, A) - self.grads[i], axes = 2)
            # add = np.tensor(self.dot(C, A) - self.grads[i]);
            g += add
        return g
    

    def compute(self, verbose):
        C = np.random.randn(self.dimC, self.d[0], self.d[1]) if self.C is None else self.C # dimC gradients of size (d[0], d[1])
        A = np.ones((self.dimC)) * 1 / self.dimC if self.A is None else self.A # starting off as probability simplex
        history = 0;
        for iter in range(self.iters):
            c_grad = self.grad_C(C, A);
            obj_value = self.objective(C, A);
            akc = (obj_value + (10/(10+iter)))/np.square(np.linalg.norm(c_grad));
            C = C - akc*c_grad
            a_grad = self.grad_A(C, A, 0);
            aka = (obj_value + (10/(10+iter)))/np.square(np.linalg.norm(a_grad));
            update = A - aka*a_grad
            A = self.projToSmplx(A - aka*a_grad);
        history = obj_value;
        self.C = C
        self.A = A;
        if(verbose):
            plt.plot(range(self.iters), history);
        return history;

    
    def quantize(self, dimC, g):
        dimC = min(dimC, self.dimC)
        index = np.random.choice(np.arange(self.dimC), p=self.A);
        val = np.square(np.linalg.norm(self.C[index, :, :]-g))
        return (index, val)



