from utils import executeNN as execute
import matplotlib.pyplot as plt
def __main__():
    iters = 1000
    nagents = 5
    m = 10000
    lr = 0.95
    sigmoid = False

    exp00 = execute(nagents=nagents, dimC=5, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid, qflag=2)
    exp0 = execute(nagents=nagents, dimC=5, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid, qflag=1)

    sigmoid = True

    exp00s = execute(nagents=nagents, dimC=5, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid, qflag=2)
    exp0s = execute(nagents=nagents, dimC=5, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid, qflag=1)

    iters = 1000
    nagents = 5
    m = 10000
    lr = 0.95
    sigmoid = False;

    exp1 = execute(nagents=nagents, dimC=5, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid, qflag=0)
    exp2 = execute(nagents=nagents, dimC=10, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid, qflag=0)
    exp3 = execute(nagents=nagents, dimC=20, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid, qflag=0)
    exp4 = execute(nagents=nagents, dimC=10, r=10, m=m, iters=iters, lr=lr, sigmoid=sigmoid)
    exp5 = execute(nagents=nagents, dimC=10, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid)
    exp6 = execute(nagents=nagents, dimC=10, r=100, m=m, iters=iters, lr=lr, sigmoid=sigmoid)


    sigmoid = True # CHANGE TO SIGMOID

    exp7 = execute(nagents=nagents, dimC=5, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid)
    exp8 = execute(nagents=nagents, dimC=10, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid)
    exp9 = execute(nagents=nagents, dimC=20, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid)
    exp10 = execute(nagents=nagents, dimC=10, r=10, m=m, iters=iters, lr=lr, sigmoid=sigmoid)
    exp11 = execute(nagents=nagents, dimC=10, r=50, m=m, iters=iters, lr=lr, sigmoid=sigmoid)
    exp12 = execute(nagents=nagents, dimC=10, r=100, m=m, iters=iters, lr=lr, sigmoid=sigmoid)

    # BATCH 1
    print(f'Compression1: {exp1[5]}')
    print(f'Quantize1: {exp1[6]}')
    print(f'Compression2: {exp2[5]}')
    print(f'Quantize2: {exp2[6]}')
    print(f'Compression3: {exp3[5]}')
    print(f'Quantize3: {exp3[6]}')

    # History plot - varying dimC, relu
    plt.figure(1)
    plt.plot(range(iters), exp1[0], label = 'M = 5')
    plt.plot(range(iters), exp2[0], label = 'M = 10')
    plt.plot(range(iters), exp3[0], label = 'M = 20')
    plt.plot(range(iters), exp0[0], label = 'Full precision baseline')
    plt.plot(range(iters), exp00[0], label = '2-bit quantization')
    plt.yscale("log")
    plt.xlabel("Iteration #")
    plt.ylabel("Relative Optimality Gap")
    plt.title("Optimality gap vs Iteration, dep. on M (dimension C) - ReLU activation")
    plt.legend()
    plt.show()

    # Compute history - varying dimC, relu
    plt.figure(1)
    plt.plot(range(len(exp1[1])), exp1[1], label = 'First layer - M = 5')
    plt.plot(range(len(exp1[2])), exp1[2], label = 'Second layer - M = 5')
    plt.plot(range(len(exp2[1])), exp2[1], label = 'First layer - M = 10')
    plt.plot(range(len(exp2[2])), exp2[2], label = 'Second layer - M = 10')
    plt.plot(range(len(exp3[1])), exp3[1], label = 'First layer - M = 20')
    plt.plot(range(len(exp3[2])), exp3[2], label = 'Second layer - M = 20')
    plt.yscale("log")
    plt.xlabel("Iteration #")
    plt.ylabel("Relative Optimality Gap")
    plt.title("Convergence of Recompute, dep. on M (dimension C) - ReLU activation")
    plt.legend()
    plt.show()

    # BATCH 2
    print(f'Compression1: {exp4[5]}')
    print(f'Quantize1: {exp4[6]}')
    print(f'Compression2: {exp5[5]}')
    print(f'Quantize2: {exp5[6]}')
    print(f'Compression3: {exp6[5]}')
    print(f'Quantize3: {exp6[6]}')

    # History plot - varying r, relu
    plt.figure(1)
    plt.plot(range(iters), exp4[0], label = 'Frq = 10')
    plt.plot(range(iters), exp5[0], label = 'Frq = 50')
    plt.plot(range(iters), exp6[0], label = 'Frq = 100')
    plt.yscale("log")
    plt.plot(range(iters), exp0[0], label = 'Full precision baseline')
    plt.plot(range(iters), exp00[0], label = '2-bit quantization')
    plt.xlabel("Iteration #")
    plt.ylabel("Relative Optimality Gap")
    plt.title("Optimality gap vs Iteration, dep. on Frequency of Recompute - ReLU activation")
    plt.legend()
    plt.show()

    # Compute history - varying r, relu
    plt.figure(1)
    plt.plot(range(len(exp4[1])), exp4[1], label = 'First layer - Frq = 10')
    plt.plot(range(len(exp4[2])), exp4[2], label = 'Second layer - Frq = 10')
    plt.plot(range(len(exp5[1])), exp5[1], label = 'First layer - Frq = 50')
    plt.plot(range(len(exp5[2])), exp5[2], label = 'Second layer - Frq = 50')
    plt.plot(range(len(exp6[1])), exp6[1], label = 'First layer - Frq = 100')
    plt.plot(range(len(exp6[2])), exp6[2], label = 'Second layer - Frq = 100')
    plt.yscale("log")
    plt.xlabel("Iteration #")
    plt.ylabel("Relative Optimality Gap")
    plt.title("Convergence of Recompute, dep. on Frequency of Recompute - ReLU activation")
    plt.legend()
    plt.show()

    # BATCH 3
    print(f'Compression1: {exp7[5]}')
    print(f'Quantize1: {exp7[6]}')
    print(f'Compression2: {exp8[5]}')
    print(f'Quantize2: {exp8[6]}')
    print(f'Compression3: {exp9[5]}')
    print(f'Quantize3: {exp9[6]}')

    # History plot - varying dimC, sigmoid
    plt.figure(1)
    plt.plot(range(iters), exp7[0], label = 'M = 5')
    plt.plot(range(iters), exp8[0], label = 'M = 10')
    plt.plot(range(iters), exp9[0], label = 'M = 20')
    plt.plot(range(iters), exp0s[0], label = 'Full precision baseline')
    plt.plot(range(iters), exp00s[0], label = '2-bit quantization')
    plt.yscale("log")
    plt.xlabel("Iteration #")
    plt.ylabel("Relative Optimality Gap")
    plt.title("Optimality gap vs Iteration, dep. on M (dimension C) - Sigmoid activation")
    plt.legend()
    plt.show()

    # Compute history - varying dimC, sigmoid
    plt.figure(1)
    plt.plot(range(len(exp7[1])), exp7[1], label = 'First layer - M = 5')
    plt.plot(range(len(exp7[2])), exp7[2], label = 'Second layer - M = 5')
    plt.plot(range(len(exp8[1])), exp8[1], label = 'First layer - M = 10')
    plt.plot(range(len(exp8[2])), exp8[2], label = 'Second layer - M = 10')
    plt.plot(range(len(exp9[1])), exp9[1], label = 'First layer - M = 20')
    plt.plot(range(len(exp9[2])), exp9[2], label = 'Second layer - M = 20')
    plt.yscale("log")
    plt.xlabel("Iteration #")
    plt.ylabel("Relative Optimality Gap")
    plt.title("Convergence of Recompute, dep. on M (dimension C) - Sigmoid activation")
    plt.legend()
    plt.show()

    # BATCH 4
    print(f'Compression1: {exp10[5]}')
    print(f'Quantize1: {exp10[6]}')
    print(f'Compression2: {exp11[5]}')
    print(f'Quantize2: {exp11[6]}')
    print(f'Compression3: {exp12[5]}')
    print(f'Quantize3: {exp12[6]}')

    # History plot - varying r, sigmoid
    plt.figure(1)
    plt.plot(range(iters), exp10[0], label = 'Frq = 10')
    plt.plot(range(iters), exp11[0], label = 'Frq = 50')
    plt.plot(range(iters), exp12[0], label = 'Frq = 100')
    plt.yscale("log")
    plt.xlabel("Iteration #")
    plt.ylabel("Relative Optimality Gap")
    plt.title("Optimality gap vs Iteration, dep. on Frequency of Recompute - Sigmoid activation")
    plt.plot(range(iters), exp0s[0], label = 'Full precision baseline')
    plt.plot(range(iters), exp00s[0], label = '2-bit quantization')
    plt.legend()
    plt.show()

    # Compute history - varying r, sigmoid
    plt.figure(1)
    plt.plot(range(len(exp10[1])), exp10[1], label = 'First layer - Frq = 10')
    plt.plot(range(len(exp10[2])), exp10[2], label = 'Second layer - Frq = 10')
    plt.plot(range(len(exp11[1])), exp11[1], label = 'First layer - Frq = 50')
    plt.plot(range(len(exp11[2])), exp11[2], label = 'Second layer - Frq = 50')
    plt.plot(range(len(exp12[1])), exp12[1], label = 'First layer - Frq = 100')
    plt.plot(range(len(exp12[2])), exp12[2], label = 'Second layer - Frq = 100')
    plt.yscale("log")
    plt.xlabel("Iteration #")
    plt.ylabel("Relative Optimality Gap")
    plt.title("Convergence of Recompute, dep. on Frequency of Recompute - Sigmoid activation")
    plt.legend()
    plt.show()