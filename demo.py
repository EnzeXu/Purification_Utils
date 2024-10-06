import numpy as np
import time

from purification import extract, purify_2d_sequential, purify_3d_sequential, purify_3d_parallel


def demo_extract():
    # Demo: extract function
    full_terms, terms, coefficient_terms = extract("3*y+2*sin(x)-3*x**2+2*x*y")
    print(f"full_terms: {full_terms}")
    print(f"terms: {terms}")
    print(f"coefficient_terms: {coefficient_terms}")
    # full_terms: [2 * sin(x), -3 * x ** 2, 2 * x * y, 3 * y]
    # terms: [sin(x), x ** 2, x * y, y]
    # coefficient_terms: [2, -3, 2, 3]

def demo_purify_2d():
    # Demo: purify function - 2D: [N, m]
    t0 = time.time()
    example_eq = "-0.006380087387799799*x + 1.0092640158560806*x/z - 0.099003265454531446*y/z + 0.3354600291292666*z - 10.33025282098863"
    example_data: np.ndarray = np.load("example_2d.npy")  # N * dim
    example_variable_list = ["x", "y", "z"]
    # You can choose either the sequential (purify) or the parallel version (purify_parallel)
    purified_eq, avg_ratio, full_terms, terms = purify_2d_sequential(example_eq, example_data, example_variable_list, threshold=0.01)
    # purified_eq, avg_ratio, full_terms, terms = purify_parallel(example_eq, example_data, example_variable_list, threshold=0.01)

    print(f"example_data shape: {example_data.shape}")
    print(f"eq: {example_eq}")
    print(f"purified_eq: {purified_eq}")
    print(f"avg_ratio: {avg_ratio}")
    print(f"full_terms: {full_terms}")
    print(f"terms: {terms}")
    print(f"Time cost: {time.time() - t0:.2f} s")
    # example_data shape: (1000, 3)
    # eq: -0.006380087387799799 * x + 1.0092640158560806 * x / z - 0.099003265454531446 * y / z + 0.3354600291292666 * z - 10.33025282098863
    # purified_eq: 1.0092640158560806*x/z - 0.099003265454531446*y/z - 10.33025282098863
    # avg_ratio: [0.17057849 0.00732359 0.77513511 0.03837589 0.00858691]
    # full_terms: [-10.33025282098863, -0.006380087387799799*x, 1.0092640158560806*x/z, -0.099003265454531446*y/z, 0.3354600291292666*z]
    # terms: [1, x, x/z, y/z, z]


def demo_purify_3d(parallel=False):
    # Demo: purify function - 3D: [n_traj, N, m]
    t0 = time.time()
    example_eq = "-0.006380087387799799*x + 1.0092640158560806*x/z - 0.099003265454531446*y/z + 0.3354600291292666*z - 10.33025282098863"
    example_data: np.ndarray = np.load("example_3d.npy")  # N * dim
    example_variable_list = ["x", "y", "z"]
    # You can choose either the sequential (purify) or the parallel version (purify_parallel)

    if parallel:
        purified_eq, avg_ratio, full_terms, terms = purify_3d_parallel(example_eq, example_data, example_variable_list, threshold=0.01)
    else:
        purified_eq, avg_ratio, full_terms, terms = purify_3d_sequential(example_eq, example_data, example_variable_list, threshold=0.01)

    print(f"example_data shape: {example_data.shape}")
    print(f"eq: {example_eq}")
    print(f"purified_eq: {purified_eq}")
    print(f"avg_ratio: {avg_ratio}")
    print(f"full_terms: {full_terms}")
    print(f"terms: {terms}")
    print(f"[parallel = {parallel}] Time cost: {time.time() - t0:.2f} s")
    # example_data shape: (1000, 3)
    # eq: -0.006380087387799799 * x + 1.0092640158560806 * x / z - 0.099003265454531446 * y / z + 0.3354600291292666 * z - 10.33025282098863
    # purified_eq: 1.0092640158560806*x/z - 0.099003265454531446*y/z - 10.33025282098863
    # avg_ratio: [0.17057849 0.00732359 0.77513511 0.03837589 0.00858691]
    # full_terms: [-10.33025282098863, -0.006380087387799799*x, 1.0092640158560806*x/z, -0.099003265454531446*y/z, 0.3354600291292666*z]
    # terms: [1, x, x/z, y/z, z]


if __name__ == "__main__":
    # demo_extract()
    # demo_purify_2d()


    # you can specify sequential (False) or parallel (True)
    demo_purify_3d(parallel=False)
    # [parallel = True] Time cost: 2.01 s
    # [parallel = False] Time cost: 11.98 s










