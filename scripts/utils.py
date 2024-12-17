import matplotlib.pyplot as plt
import numpy as np
from models.agent import SimplexDLSQ, EllipsoidDLSQ, NN
## Execute DLSQ without Quantization ##
def executeSimplex(nagents, dimC, r, m=10, n=10, iters=1000, lr=0.1, verbose=False, qflag=0): 
    dlsqexe = SimplexDLSQ(m, n, nagents, lr, r, dimC, qflag);
    history, compute_history, quantize_history, bit_history = dlsqexe.run(iters, True)
    comp_ratio = np.sum(bit_history)/(n*nagents*len(history));
    print("Compression ratio achieved = ", comp_ratio);
    print("Optimal = ", dlsqexe.optimal)
    print("")
    print("Done...")
    print("___________________________________________________________________")
    if(verbose):
        plt.figure(1);
        plt.yscale('log');
        plt.plot(range(len(history)), history);
        plt.figure(2);
        plt.yscale('log');
        plt.plot(range(len(compute_history)), compute_history);
        plt.figure(3);
        plt.yscale('log');
        plt.plot(range(len(quantize_history)), quantize_history);
        plt.figure(4);
        plt.yscale('log');
        plt.plot(range(len(bit_history)), bit_history);
    return history, compute_history, quantize_history, bit_history, comp_ratio

## Execute DLSQ without Quantization ##
def executeEllispoid(nagents, dimC, r, m=10, n=10, iters=1000, lr=0.1, verbose=False, qflag=0): 
    dlsqexe = EllipsoidDLSQ(m, n, nagents, lr, r, dimC, qflag);
    history, compute_history, quantize_history, bit_history = dlsqexe.run(iters, True)
    comp_ratio = np.sum(bit_history)/(n*nagents*len(history));
    print("Compression ratio achieved = ", comp_ratio);
    print("Optimal = ", dlsqexe.optimal)
    print("")
    print("Done...")
    print("___________________________________________________________________")
    if(verbose):
        plt.figure(1);
        plt.yscale('log');
        plt.plot(range(len(history)), history);
        plt.figure(2);
        plt.yscale('log');
        plt.plot(range(len(compute_history)), compute_history);
        plt.figure(3);
        plt.yscale('log');
        plt.plot(range(len(quantize_history)), quantize_history);
        plt.figure(4);
        plt.yscale('log');
        plt.plot(range(len(bit_history)), bit_history);
    return history, compute_history, quantize_history, bit_history, comp_ratio
    
## Execute NN without Quantization ##
def executeNN(nagents, dimC, r, m=10, n=100, iters=1000, lr=1.0, sigmoid = False, qflag = 0): 
    nnexe = NN(m, nagents, lr, r, dimC, sigmoid, qflag)
    history, chistory1, chistory2, qhistory1, qhistory2, bhistory, flaghistory = nnexe.run(iters, True) # return bhistory
    compression = sum(bhistory) / (nagents * iters * (2*8 + 8*4)) # Need to create variable for weight matrices
    # print(f'Compression ratio: {compression}')
    # print(f'Quantized Iterations Ratio: {flaghistory}')
    # # compression ratio = sum(bithistory)/(nagents*32*8*num_iters)
    # print("Optimal = ", nnexe.optimal)
    # print("")
    # print("Done...")
    # print("___________________________________________________________________")
    return history, chistory1, chistory2, qhistory1, qhistory2, compression, flaghistory / iters