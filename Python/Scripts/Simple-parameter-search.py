import numpy as np
from scipy.stats import entropy

def binary_entropy(p):
    """Compute binary entropy H(p) = -p log2(p) - (1-p) log2(1-p)"""
    if 0 < p < 1:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    elif p == 0 or p == 1:
        return 0
    else:
        raise ValueError("Input to binary_entropy must be in [0, 1]")

def convolve(p, q):
    """Convolve probabilities: c = p(1-q) + q(1-p)"""
    return p * (1 - q) + q * (1 - p)

def R_GV(delta, w):
    """Compute the R_GV term"""
    Pxxtc = np.array([[1 - w - delta / 2, delta / 2],
                      [delta / 2, w - delta / 2]])
    Pxxtc = Pxxtc.flatten()  # Flatten to a 1D array for entropy calculation
    Hxxtc = entropy(Pxxtc, base=2)
    return 2 * binary_entropy(w) - Hxxtc

def GV_bound(delta):
    """Compute the Gilbert-Varshamov bound"""
    return 1 - binary_entropy(delta)

# Parameters
delta_values = np.linspace(0.001, 0.5, 100)  # Range of delta values
num_w_points = 20  # Number of w points per delta
results = []  # Store (delta, w, R) tuples

for delta in delta_values:
    jr = (1 - np.sqrt(1 - 2 * delta)) / 2
    w_values = np.linspace(jr, 0.5, num_w_points)
    
    for w in w_values:
        # Compute the rate R = Rc + Rs
        Rc = 1 - binary_entropy(convolve(delta, w))
        Rs = R_GV(delta, w)
        rate = Rc + Rs
        
        # Check if this rate is valid (greater than GV bound)
        if rate >= GV_bound(delta):
            results.append((delta, w, rate, Rc, Rs))

# Print valid (delta, w, R) tuples
for delta, w, rate, rs, rc in results:
    print(f"delta: {delta:.4f}, w: {w:.4f}, R: {rate:.4f}, Rc: {rc:.4f}, Rs: {rs:.4f}")
