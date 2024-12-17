## Distributed Least Squares Toy Model ##

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from multiprocessing import Pool
import time
from multiprocessing import Process 
from models.quantizer import *

class SimplexAgentLSQ():
    def __init__(self, A, b, nagents, r, dimC, qflag):
        self.A = A
        self.b = b
        self.r = r;
        self.nagents = nagents;
        self.grads = [];
        self.PASS = True;
        iters = 10000;
        self.Q = SimplexQuantizer(iters, dimC);
        self.qflag = qflag;
    
    def grad(self, x, flag):
        lmbda = 0.1
        g = self.A.T@self.A@x - self.A.T@self.b + lmbda*2*x
        self.grads.append(np.expand_dims(g, 1));
        ch = -1;
        qh = -1;
        bits = 0;
        if(self.qflag != 0): ## do not run if quantize
            flag = False;
        if(((len(self.grads)-self.r >= 0) or (len(self.grads) >= 2 and self.PASS)) and flag):
            G = np.hstack(self.grads);
            self.Q.load(G);
            start = time.time();
            ch = self.Q.compute(False);
            bits += self.Q.d * self.Q.dimC;
            print("Time to compute C = ", time.time() - start);
            self.grads = [];
            self.PASS = False;
            print("Recomputing C... | Converged with alpha value = ", ch)
        if(self.Q.C is not None and flag):
            bits += self.Q.dimC * (1/16);
            ghat, qh = self.Q.quantize(g) #self.Q.C.shape[1]
            return (self.Q.C[:,ghat], ch, qh, bits)
        else:
            if(self.qflag == 2):
                g = np.sign(g);
            print("Sending full precision gradients")
            bits += len(g)
            return (g, ch, qh, bits)

        
    def objective(self, x):
        return np.square(np.linalg.norm(self.A@x-self.b)) + np.square(np.linalg.norm(x));
    
class SimplexDLSQ():
    def __init__(self, m, n, nagents, lr, r, dimC, qflag):
        A = np.random.random((m,n))
        z = np.random.random((n,))
        b = A@z + np.random.random((m))

        print('Computing optimal...');
        self.optimal = np.square(np.linalg.norm(A@np.linalg.inv(A.T@A + 0.1*np.identity(n))@A.T@b-b))
        print('finished! | optimal = ', self.optimal)
        
        self.x = np.zeros((n,));
        self.m = m;
        self.n = n;
        self.r = r;
        self.nagents = nagents;
        self.lr = lr;  
        self.agents = self.distribute(A, b, nagents, r, dimC, qflag);
        
        self.A = A;
        self.b = b;
    
    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def distribute(self, A, b, nagents, r, dimC, qflag):
        agents = [];
        splits = list(self.split(range(self.m), self.nagents))
        
        for sp in splits:
            agents.append(SimplexAgentLSQ(A[sp, :], b[sp], self.nagents, r, dimC, qflag));
        return agents;
    
    def computeGradFromLocal(self, x, flag):
        temp = np.zeros((self.n,))
        normalizing = 0;
        total_bits = 0;
        for n in range(self.nagents):
            localgrad, ch, qh, bits = self.agents[n].grad(self.x, flag)
            total_bits += bits;
            temp += localgrad
            normalizing += np.linalg.norm(localgrad)
        
        return (temp/normalizing, ch, qh, total_bits);
    
    def step(self, i, flag):
        g, ch, qh, bits = self.computeGradFromLocal(self.x, flag);
        ak = (self.objective()-self.optimal)/np.square(np.linalg.norm(g));
        xnext = self.x - min(ak, self.lr/np.sqrt(i+1))*g
        self.x = xnext + 0.1*(xnext - self.x);
        return (ch, qh, bits);
    
    def objective(self):
        return np.sum([self.agents[i].objective(self.x) for i in range(self.nagents)])
        
    def run(self, iters, verbose):
        history = [];
        compute_history = [];
        quantize_history = [];
        bit_history = [];
        i = 0;
        run = True;
        while(run and i < iters):
            print("Current objective value = ", np.square(np.linalg.norm(self.A@self.x - self.b)) + np.square(np.linalg.norm(self.x)))
            gap = (self.objective()-self.optimal)/self.optimal
            print("Iteration = " + str(i) + " | Relative Optimality gap = " + str(gap))
            print("")
            if(np.abs(gap) <= 1e-1):
                run = False;
            history.append(gap);
            
            flag = True
            if(i >= self.r):
                flag = (np.diff(history[-3::], 1, 0) >= 0).all()
                flag = not flag
                
            ch, qh, bits = self.step(i, flag);
            bit_history.append(bits);
            if(ch != -1):
                compute_history.append(ch)
            if(qh != -1):
                quantize_history.append(qh);
            i += 1;
        #if(verbose):
        #    plt.yscale('log')
        #    plt.plot(range(i), history);
        return (history, compute_history, quantize_history, bit_history);


## Distributed Least Squares Toy Model ##
class EllipsoidAgentLSQ():
    def __init__(self, A, b, nagents, r, dimC, qflag):
        self.A = A
        self.b = b
        self.r = r;
        self.nagents = nagents;
        self.grads = [];
        self.PASS = True;
        iters = 10000;
        self.Q = EllipsoidQuantizer(iters, dimC);
        self.qflag = qflag;
    
    def grad(self, x, flag):
        lmbda = 0.1
        g = self.A.T@self.A@x - self.A.T@self.b + lmbda*2*x
        self.grads.append(np.expand_dims(g, 1));
        ch = -1;
        qh = -1;
        bits = 0;
        if(self.qflag != 0): ## do not run if quantize
            flag = False;
        if(((len(self.grads)-self.r >= 0) or (len(self.grads) >= 2 and self.PASS)) and flag):
            G = np.hstack(self.grads);
            self.Q.load(G);
            start = time.time();
            ch = self.Q.compute(False);
            bits += self.Q.d * self.Q.dimC;
            print("Time to compute C = ", time.time() - start);
            self.grads = [];
            self.PASS = False;
            print("Recomputing C... | Converged with alpha value = ", ch)
        if(self.Q.C is not None and flag):
            bits += self.Q.dimC * (1/16);
            ghat, qh = self.Q.quantize(g) #self.Q.C.shape[1]
            return (self.Q.C[:,ghat], ch, qh, bits)
        else:
            if(self.qflag == 2):
                g = np.sign(g);
            print("Sending full precision gradients")
            bits += len(g)
            return (g, ch, qh, bits)

        
    def objective(self, x):
        return np.square(np.linalg.norm(self.A@x-self.b)) + np.square(np.linalg.norm(x));
    
class EllipsoidDLSQ():
    def __init__(self, m, n, nagents, lr, r, dimC, qflag):
        A = np.random.random((m,n))
        z = np.random.random((n,))
        b = A@z + np.random.random((m))

        print('Computing optimal...');
        self.optimal = np.square(np.linalg.norm(A@np.linalg.inv(A.T@A + 0.1*np.identity(n))@A.T@b-b))
        print('finished! | optimal = ', self.optimal)
        
        self.x = np.zeros((n,));
        self.m = m;
        self.n = n;
        self.r = r;
        self.nagents = nagents;
        self.lr = lr;  
        self.agents = self.distribute(A, b, nagents, r, dimC, qflag);
        
        self.A = A;
        self.b = b;
    
    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

    def distribute(self, A, b, nagents, r, dimC, qflag):
        agents = [];
        splits = list(self.split(range(self.m), self.nagents))
        
        for sp in splits:
            agents.append(EllipsoidAgentLSQ(A[sp, :], b[sp], self.nagents, r, dimC, qflag));
        return agents;
    
    def computeGradFromLocal(self, x, flag):
        temp = np.zeros((self.n,))
        normalizing = 0;
        total_bits = 0;
        for n in range(self.nagents):
            localgrad, ch, qh, bits = self.agents[n].grad(self.x, flag)
            total_bits += bits;
            temp += localgrad
            normalizing += np.linalg.norm(localgrad)
        
        return (temp/normalizing, ch, qh, total_bits);
    
    def step(self, i, flag):
        g, ch, qh, bits = self.computeGradFromLocal(self.x, flag);
        ak = (self.objective()-self.optimal)/np.square(np.linalg.norm(g));
        xnext = self.x - min(ak, self.lr/np.sqrt(i+1))*g
        self.x = xnext + 0.1*(xnext - self.x);
        return (ch, qh, bits);
    
    def objective(self):
        return np.sum([self.agents[i].objective(self.x) for i in range(self.nagents)])
        
    def run(self, iters, verbose):
        history = [];
        compute_history = [];
        quantize_history = [];
        bit_history = [];
        i = 0;
        run = True;
        while(run and i < iters):
            print("Current objective value = ", np.square(np.linalg.norm(self.A@self.x - self.b)) + np.square(np.linalg.norm(self.x)))
            gap = (self.objective()-self.optimal)/self.optimal
            print("Iteration = " + str(i) + " | Relative Optimality gap = " + str(gap))
            print("")
            if(np.abs(gap) <= 1e-1):
                run = False;
            history.append(gap);
            
            flag = True
            if(i >= self.r):
                flag = (np.diff(history[-3::], 1, 0) >= 0).all()
                flag = not flag
                
            ch, qh, bits = self.step(i, flag);
            bit_history.append(bits);
            if(ch != -1):
                compute_history.append(ch)
            if(qh != -1):
                quantize_history.append(qh);
            i += 1;
        #if(verbose):
        #    plt.yscale('log')
        #    plt.plot(range(i), history);
        return (history, compute_history, quantize_history, bit_history);

# Sample Neural Network with 2 Layers -> FC (256 x 1024) - ReLU - FC (1024 x 10)
class AgentNN():
    def __init__(self, x, y, nagents, r, dimC, sigmoid = False, qflag=0):
        self.x = x
        self.y = y
        self.nagents = nagents
        self.r = r
        self.grads1 = []
        self.grads2 = []
        self.PASS = True
        iters = 100
        self.Q1 = NNQuantizer(iters, dimC) # Create a new Quantizer for matrices - first layer
        self.Q2 = NNQuantizer(iters, dimC) # Create a new Quantizer for matrices - second layer
        self.sigmoid = sigmoid
        self.qflag = qflag;

    def sigmoid_func(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def get_h(self, W1):
        if self.sigmoid:
            return self.sigmoid_func(self.x @ W1)
        return np.maximum(0, self.x @ W1)

    def grad(self, W1, W2, flag):
        # Computer dW2 and dW1
        h = self.get_h(W1)
        dW2 = h.T @ (h @ W2 - self.y)
        self.grads2.append(dW2)

        dh = (h @ W2 - self.y) @ W2.T
        dW1 = self.x.T @ (dh * (h > 0)) # Check this, I think this is right
        if self.sigmoid:
            # dW1 = self.x.T @ (h.T @ (1 - h)) @ dh
            dW1 = self.x.T @ (dh * h * (1 - h))
        self.grads1.append(dW1)

        ch1 = -1
        ch2 = -1
        qh1 = -1
        qh2 = -1
        bits = 0
        if(self.qflag != 0):
          flag = False;

        if (((len(self.grads1) % self.r == 0 and len(self.grads1) != 0) or (len(self.grads1) >= 2 and self.PASS)) and flag):
            G1 = np.zeros((len(self.grads1), self.grads1[0].shape[0], self.grads1[0].shape[1]))
            G2 = np.zeros((len(self.grads2), self.grads2[0].shape[0], self.grads2[0].shape[1]))
            for k in range(len(self.grads1)):
                G1[k, :, :] = self.grads1[k]
                G2[k, :, :] = self.grads2[k]

            self.Q1.load(G1)
            self.Q2.load(G2)

            ch1 = self.Q1.compute(False)
            ch2 = self.Q2.compute(False) # Is there a way to combine ch1 and ch2
            bits += self.Q1.dimC * self.grads1[0].shape[0] * self.grads1[0].shape[1]
            bits += self.Q2.dimC * self.grads2[0].shape[0] * self.grads2[0].shape[1]
            self.grads1 = []
            self.grads2 = []

            self.PASS = False
            print("Recomputing C for layer 1... | Converged with value = ", ch1)
            print("Recomputing C for layer 2... | Converged with value = ", ch2)
        if self.Q1.C is not None and self.Q2.C is not None and flag:
            bits += self.Q1.dimC + self.Q2.dimC # dimC is the same for both Q1 and Q2

            # This needs to be changed to fit the matrix gradient definition
            g1, qh1 = self.Q1.quantize(self.Q1.C.shape[0], dW1)
            g2, qh2 = self.Q2.quantize(self.Q2.C.shape[0], dW2)

            return (self.Q1.C[g1], self.Q2.C[g2], ch1, ch2, qh1, qh2, bits) # return bits here
        else:
            bits += dW1.shape[0]*dW1.shape[1] + dW2.shape[0]*dW2.shape[1] 
            print("Sending full precision gradient")
            if(self.qflag == 2):
              dW1 = np.sign(dW1)
              dW2 = np.sign(dW2)
            return (dW1, dW2, ch1, ch2, qh1, qh2, bits) # return bits here

    def objective(self, W1, W2):
        h = np.maximum(0, self.x @ W1)
        if self.sigmoid:
            h = self.sigmoid_func(self.x @ W1)
        return np.square(np.linalg.norm(h @ W2 - self.y))


class NN():
    def __init__(self, m, nagents, lr, r, dimC, sigmoid = False, qflag=0):
        w1_in = 2
        w2_in = 8
        w_out = 4
        x = np.random.randn(m, w1_in)
        z = np.random.random((w1_in, w2_in)) # Proxy for W1 - not used
        w = np.random.random((w2_in, w_out)) # Proxy for W2 - not used
        y = np.maximum(0, x @ z) @ w;
        self.sigmoid = sigmoid
        if sigmoid:
            y = self.sigmoid_func(x @ z) @ w

        self.optimal = 0 # Not adding noise for now
        
        self.W1 = np.random.rand(w1_in, w2_in) # Not fixed
        self.W2 = np.random.rand(w2_in, w_out) # Not fixed
        self.h = np.maximum(0, x @ self.W1) # Not fixed - Intermediate output after ReLU
        if sigmoid:
            self.h = self.sigmoid_func(x @ z)

        self.m = m
        self.nagents = nagents # Each layers will have this many agents
        self.lr = lr
        self.r = r

        self.agents = self.distribute(x, y, nagents, r, dimC, sigmoid, qflag)
        self.x = x # Fixed
        self.y = y # Fixed


    # def forward(self, update = True):
    #     h = np.maximum(0, self.x @ self.W1)
    #     pred_y = h @ self.W2
    #     self.h = h if update
    #     return pred_y


    def split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    

    def distribute(self, x, y, nagents, r, dimC, sigmoid, qflag):
        agents = [];
        splits = list(self.split(range(self.m), self.nagents))
        
        for sp in splits:
            agents.append(AgentNN(x[sp, :], y[sp, :], self.nagents, r, dimC, sigmoid, qflag))
        return agents


    def computeGradFromLocal(self, W1, W2, flag):
        temp1 = np.zeros_like(W1)
        norm1 = 0
        temp2 = np.zeros_like(W2)
        norm2 = 0
        total_bits = 0

        for i in range(self.nagents):
            localgrad1, localgrad2, ch1, ch2, qh1, qh2, bits = self.agents[i].grad(W1, W2, flag) # return bits from grad
            temp1 += localgrad1
            temp2 += localgrad2
            norm1 += np.linalg.norm(localgrad1)
            norm2 += np.linalg.norm(localgrad2)
            total_bits += bits

        return (temp1 / norm1 / self.nagents, temp2 / norm2 / self.nagents, ch1, ch2, qh1, qh2, total_bits) # return total_bits


    def sigmoid_func(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


    def step(self, i, flag):
        g1, g2, ch1, ch2, qh1, qh2, total_bits = self.computeGradFromLocal(self.W1, self.W2, flag)
        self.W1 = self.W1 - self.lr / np.sqrt(i + 1) * g1
        self.W2 = self.W2 - self.lr / np.sqrt(i + 1) * g2
        return (ch1, ch2, qh1, qh2, total_bits)


    def objective(self):
        return np.sum([self.agents[i].objective(self.W1, self.W2) for i in range(self.nagents)])

    
    def run(self, iters, verbose):
        history = []
        chistory1 = []
        chistory2 = []
        qhistory1 = []
        qhistory2 = []
        i = 0
        run = True
        bhistory = []
        flaghistory = 0

        while run and i < iters:
            h = np.maximum(0, self.x @ self.W1)
            if self.sigmoid:
                h = self.sigmoid_func(self.x @ self.W1)
            print("Current objective value = ", self.objective()) # Try using provided objective function
            gap = (self.objective() - self.optimal)
            print("Iteration = " + str(i) + " | Relative Optimality gap = " + str(gap))
            print("")

            if np.abs(gap) <= 1e-1:
                run = False
            history.append(gap)

            flag = True
            if (i >= self.r):
                flag = (np.diff(history[-3::], 1, 0) >= 0).all()
                flag = not flag
            flaghistory += (flag)


            ch1, ch2, qh1, qh2, total_bits = self.step(i, flag) # return total_bits
            bhistory.append(total_bits)
            if ch1 != -1:
                chistory1.append(ch1)

            if ch2 != -1:
                chistory2.append(ch2)

            if qh1 != -1:
                qhistory1.append(qh1)

            if qh2 != -1:
                qhistory2.append(qh2)

            i += 1

        if verbose:
            plt.yscale('log')
            plt.plot(range(i), history)

        return (history, chistory1, chistory2, qhistory1, qhistory2, bhistory, flaghistory)