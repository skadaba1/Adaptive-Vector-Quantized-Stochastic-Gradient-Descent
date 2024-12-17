from utils import executeEllispoid as execute
import matplotlib.pyplot as plt
import numpy as np
import scipy
from models.quantizer import EllipsoidQuantizer as Quantizer
def __main__():
        # No quantization here #
    h1, c1, q1, b1, cr1 = execute(nagents=5, dimC=5, r=50, m=10000, n=50, iters=1000, lr=0.1, verbose=False, qflag=1)
    h2, c2, q2, b2, cr2 = execute(nagents=5, dimC=5, r=50, m=10000, n=50, iters=1000, lr=0.1, verbose=False, qflag=2)

    exp1 = execute(nagents=5, dimC=5, r=100, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    exp2 = execute(nagents=5, dimC=10, r=100, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    exp3 = execute(nagents=5, dimC=20, r=100, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    print(exp1[4], exp2[4], exp3[4])
    plt.yscale("log")
    index = 0; #history, compute_history, quantize_history, bit_history, comp_ratio
    plt.figure(1)
    plt.xlabel("Iteration #");
    plt.ylabel("Relative Optimality Gap")
    plt.title("Optimality gap vs Iteration, dep. on M (dimension C)")
    plt.plot(range(len(exp1[index])), exp1[index], label="M = 5")
    plt.plot(range(len(exp2[index])), exp2[index], label="M = 10")
    plt.plot(range(len(exp3[index])), exp3[index], label="M = 20")
    plt.plot(range(len(h2)), h2, label='2-bit quantization')
    plt.plot(range(len(h1)), h1, label='Full precision baseline')
    plt.legend()

    index = 1; 
    plt.figure(2)
    plt.title("Convergence of Recompute, dep. on M (dimension C)")
    plt.xlabel("Iteration #");
    plt.ylabel("Minimum euclidean distance to true gradient")
    plt.plot(range(len(exp1[index])), exp1[index], label="M = 5")
    plt.plot(range(len(exp2[index])), exp2[index], label="M = 10")
    plt.plot(range(len(exp3[index])), exp3[index], label="M = 20")
    plt.legend()

    ## Plot Ellipse for Presentation Figures ##
    m, n = 10, 2
    A = np.random.random((m,n))
    z = np.random.random((n,))
    b = A@z + np.random.random((m))
    x = np.zeros((n,));
    grads = []

    r = 20;
    iters = r
    dimC = 5;
    Q = Quantizer(iters, dimC);
    ak = 0.01;
    for i in range(r):
        g = A.T@A@x - A.T@b;
        x = x - ak*g + np.random.randn(n)
        grads.append(np.expand_dims(g, 1));
    G = np.hstack(grads);

    Q.load(G)
    Q.compute(False)
    Q.plot(G)

    plt.figure()
    G = G.T
    hull = scipy.spatial.ConvexHull(G)
    import matplotlib.pyplot as plt
    plt.plot(G[:,0], G[:,1], 'o')
    for simplex in hull.simplices:
        plt.plot(G[simplex, 0], G[simplex, 1], 'k-')

    exp4 = execute(nagents=5, dimC=20, r=100, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    exp5 = execute(nagents=10, dimC=20, r=100, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    exp6 = execute(nagents=50, dimC=20, r=100, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    print(exp4[4], exp5[4], exp6[4])
    plt.yscale("log")
    index = 0; #history, compute_history, quantize_history, bit_history, comp_ratio
    plt.figure(1)
    plt.xlabel("Iteration #");
    plt.ylabel("Relative Optimality Gap")
    plt.title("Optimality gap vs Iteration, dep. on # Agents")
    plt.plot(range(len(exp4[index])), exp4[index], label="n_agents = 5")
    plt.plot(range(len(exp5[index])), exp5[index], label="n_agents = 10")
    plt.plot(range(len(exp6[index])), exp6[index], label="n_agents = 50")
    plt.plot(range(len(h2)), h2, label='2-bit quantization')
    plt.plot(range(len(h1)), h1, label='Full precision baseline')
    plt.legend()

    index = 1; 
    plt.figure(2)
    plt.title("Convergence of Recompute, dep. on # Agents")
    plt.xlabel("Iteration #");
    plt.ylabel("Minimum euclidean distance to true gradient")
    plt.plot(range(len(exp4[index])), exp4[index], label="n_agents = 5")
    plt.plot(range(len(exp5[index])), exp5[index], label="n_agents = 10")
    plt.plot(range(len(exp6[index])), exp6[index], label="n_agents = 50")
    plt.legend()

    exp7 = execute(nagents=5, dimC=20, r=10, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    exp8 = execute(nagents=5, dimC=20, r=50, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    exp9 = execute(nagents=5, dimC=20, r=100, m=10000, n=50, iters=1000, lr=0.1, verbose=False)
    print(exp7[4], exp8[4], exp9[4])
    plt.yscale("log")
    index = 0; #history, compute_history, quantize_history, bit_history, comp_ratio
    plt.figure(1)
    plt.xlabel("Iteration #");
    plt.ylabel("Relative Optimality Gap")
    plt.title("Optimality gap vs Iteration, dep. on Frequency of Recompute")
    plt.plot(range(len(exp7[index])), exp7[index], label="Frq = 10")
    plt.plot(range(len(exp8[index])), exp8[index], label="Frq = 50")
    plt.plot(range(len(exp9[index])), exp9[index], label="Frq = 100")
    plt.plot(range(len(h2)), h2, label='2-bit quantization')
    plt.plot(range(len(h1)), h1, label='Full precision baseline')
    plt.legend()

    index = 1; 
    plt.figure(2)
    plt.title("Convergence of Recompute, dep. on Frequency")
    plt.xlabel("Iteration #");
    plt.ylabel("Minimum euclidean distance to true gradient")
    t1 = [2*i for i in range(len(exp8[index]))]
    t2 = [10*i for i in range(len(exp9[index]))]
    plt.plot(range(len(exp7[index])), exp7[index], label="Frq = 10")
    plt.plot(t1, exp8[index], label="Frq = 50")
    plt.plot(t2, exp9[index], label="Frq = 100")
    plt.legend()

if __name__ == "__main__":
    __main__()