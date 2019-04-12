from matplotlib.colors import rgb2hex
import numpy as np


def random_rbg(x, ns):
    if ns == 0:
        np.random.seed(50)
    else:
        rand_int = int(np.random.uniform(0,100))
        print("RANDOM SEED GENERATED: " + str(rand_int))
        np.random.seed(rand_int)
    rbg_vals = np.random.uniform(size=x*3)
    A = rbg_vals.reshape((x,3))
    hex_vals = [rgb2hex(A[i,:]) for i in range(A.shape[0])]
    return hex_vals


# rs 9,63 wasn't bad
