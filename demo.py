import numpy as np

from purification import extract, purify

if __name__ == "__main__":
    # Demo: extract function
    full_terms, terms, coefficient_terms = extract("3*y+2*sin(x)-3*x**2+2*x*y")
    print(f"full_terms: {full_terms}")
    print(f"terms: {terms}")
    print(f"coefficient_terms: {coefficient_terms}")
    # full_terms: [2 * sin(x), -3 * x ** 2, 2 * x * y, 3 * y]
    # terms: [sin(x), x ** 2, x * y, y]
    # coefficient_terms: [2, -3, 2, 3]

    # Demo: purify function
    example_eq = "-0.006380087387799799*x + 1.0092640158560806*x/z - 0.099003265454531446*y/z + 0.3354600291292666*z - 10.33025282098863"
    example_data: np.ndarray = np.load("example.npy")  # N * dim
    example_variable_list = ["x", "y", "z"]
    purified_eq, avg_ratio, full_terms, terms = purify(example_eq, example_data, example_variable_list, threshold=0.01)
    print(f"example_data shape: {example_data.shape}")
    print(f"eq: {example_eq}")
    print(f"purified_eq: {purified_eq}")
    print(f"avg_ratio: {avg_ratio}")
    print(f"full_terms: {full_terms}")
    print(f"terms: {terms}")
    # example_data shape: (1000, 3)
    # eq: -0.006380087387799799 * x + 1.0092640158560806 * x / z - 0.099003265454531446 * y / z + 0.3354600291292666 * z - 10.33025282098863
    # purified_eq: 1.0092640158560806*x/z - 0.099003265454531446*y/z - 10.33025282098863
    # avg_ratio: [0.17057849 0.00732359 0.77513511 0.03837589 0.00858691]
    # full_terms: [-10.33025282098863, -0.006380087387799799*x, 1.0092640158560806*x/z, -0.099003265454531446*y/z, 0.3354600291292666*z]
    # terms: [1, x, x/z, y/z, z]

