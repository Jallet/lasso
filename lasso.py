#!/usr/bin/env python
import numpy as np
np.random.seed(100)
import argparse
import matplotlib.pyplot as plt

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--col-dim",
            help = "column dimension", 
            action = "store", type = int,
            dest = "col_dim", default = 10)
    parser.add_argument("-r", "--row-dim",
            help = "row dimension", 
            action = "store", type = int,
            dest = "row_dim", default = 10)
    parser.add_argument("--ratio", 
            help = "ratio of L1 Norm", 
            action = "store", type = float, 
            dest = "ratio", default= 0.5)
    parser.add_argument("--lr", 
            help = "learning rate",
            action = "store", type = float,
            dest = "lr", default = 0.01)
    parser.add_argument("--iters",
            help = "iterations",
            action = "store", type = int,
            dest = "iters", default = 300)
    return parser

def proximal(A, x,  b, ratio, base_lr, iters):
    history_x = []
    history_fx = []
    for it in range(iters):
        history_x.append(x)
        lr = base_lr
        # lr = 100 * base_lr / (1 + it)
        # print "lr: ", lr
        # calculate loss
        # print A[0][0], " * ", x[0][0], " + ", b[0][0]
        a = A * x - b
        least_square = np.sum(np.multiply(a, a)) 
        l1 = np.sum(np.fabs(x))
        fx = 0.5 * least_square + ratio * l1
        history_fx.append(fx)
        # print "iter: ", it, " fx: ", fx
        # print fx
        # update x
        AT = A.transpose()
        x1 = x - lr * AT * (A * x - b)
        # print "x1: ", x1
        t = np.fabs(x1) - ratio * lr
        # print "t: ", t
        x = np.multiply(np.sign(x1), np.multiply((t > 0), t))
    return history_x, history_fx

def subgradient(A, x, b, ratio, base_lr, iters):
    # print "subgradient"
    history_x = []
    history_fx = []
    for it in range(iters):
        history_x.append(x)
        a = A * x - b
        least_square = np.sum(np.multiply(a, a)) 
        l1 = np.sum(np.fabs(x))
        fx = 0.5 * least_square + ratio * l1
        history_fx.append(fx)
        # print "iter: ", it, " fx: ", fx
        # print fx
        lr = base_lr
        AT = A.transpose()
        x = x - lr * (AT * (A * x - b) + ratio * np.sign(x))
    return history_x, history_fx

def sgd(A, x, b, ratio, base_lr, iters):
    # print "sgd"
    history_x = []
    history_fx = []
    for it in range(iters):
        history_x.append(x)
        a = A * x - b
        least_square = np.sum(np.multiply(a, a)) 
        l1 = np.sum(np.fabs(x))
        fx = 0.5 * least_square + ratio * l1
        history_fx.append(fx)
        # print "iter: ", it, " fx: ", fx
        # print fx
        lr = base_lr
        sample = np.random.randint(A.shape[0])
        A1 = A[sample, :]
        b1 = b[sample]
        AT = A1.transpose()
        x = x - lr * (AT * (A1 * x - b1) + ratio * np.sign(x))
    return history_x, history_fx
         



def admm(A, x, b, ratio, base_lr, iters):
    # print "admm"
    # y = np.asmatrix(np.random.rand(x.shape[0]))
    # v = np.asmatrix(np.random.rand(x.shape[0]))
    history_x = []
    history_fx = []
    y = np.asmatrix(np.array([0]))
    v = np.asmatrix(np.array([0]))
    I = np.asmatrix(np.identity(A.shape[1]))
    
    # print "x: ", x
    # print "y: ", y
    # print "v: ", v

    
    for it in range(iters):
        history_x.append(x)
        a = A * x - b
        least_square = np.sum(np.multiply(a, a)) 
        l1 = np.sum(np.fabs(x))
        fx = 0.5 * least_square + ratio * l1
        history_fx.append(fx)
        # print "iter: ", it, " fx: ", fx
        # print fx
        lr = base_lr
        # lr = 100 * base_lr / (1 + it)
        c = lr
        AT = A.transpose()
        x = np.linalg.inv(AT * A + c * I) * (c * y - v + AT * b)
        # print "x: ", x
        z = x + v / c
        t = np.fabs(z) - ratio / c
        y = np.multiply(np.sign(z), np.multiply((t > 0), t))
        # print "y: ", y
        v = v + c * (x - y)
        # print "v: ", v
    return history_x, history_fx


def main():
    parser = argparser()
    args = parser.parse_args()
    col_dim = args.col_dim
    row_dim = args.row_dim
    ratio = args.ratio
    lr = args.lr
    iters = args.iters
    A = np.asmatrix(np.random.rand(col_dim, row_dim))
    init_x = np.asmatrix(np.random.rand(row_dim, 1))
    b = np.asmatrix(np.random.rand(col_dim, 1))
    # A = np.asmatrix(np.array([1]))
    # init_x = np.asmatrix(np.array([100]))
    # b = np.asmatrix(np.array([1]))
    print "A:"
    print A
    print "b:"
    print b
    print "init_x: "
    print init_x
    solve = proximal
    [proximal_x,  proximal_fx] = solve(A, init_x, b, ratio, lr, iters)
    solve = admm
    [admm_x,  admm_fx] = solve(A, init_x, b, ratio, lr, iters)
    solve = subgradient
    [subgradient_x,  subgradient_fx] = solve(A, init_x, b, ratio, lr, iters)
    solve = sgd
    [sgd_x,  sgd_fx] = solve(A, init_x, b, ratio, lr, iters)
    
    print np.asarray(proximal_x).shape
    print np.asarray(proximal_fx).shape
    # proximal_history = np.hstack((np.asarray(proximal_x), np.asarray(proximal_fx)))
    # admm_history = np.hstack((np.asarray(admm_x), np.asarray(admm_fx)))
    # sgd_history = np.hstack((np.asarray(sgd_x), np.asarray(sgd_fx)))
    # subgradient_history = np.hstack((np.asarray(subgradient_x), np.asarray(subgradient_fx)))

    np.savetxt("proximal", proximal_fx, fmt = "%f")
    np.savetxt("admm", admm_fx, fmt = "%f")
    np.savetxt("sgd", sgd_fx, fmt = "%f")
    np.savetxt("subgradient", subgradient_fx, fmt = "%f")

    plt.figure()
    plt.plot(proximal_fx, "g-", label = "proximal")
    plt.plot(admm_fx, "r-", label = "admm")
    plt.plot(sgd_fx, "b-", label = "sgd")
    plt.plot(subgradient_fx, "y-", label = "subgradient")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # print "min_x = ", min_x
    # print "min_fx = ", min_fx
    return 0

if __name__ == "__main__":
    main()
