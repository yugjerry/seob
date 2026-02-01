
import nashpy as nash
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt

def kl_divergence(observed, predicted):
    return entropy(observed, predicted)

def expected_payoff(U, opponent_strategy, axis=1):
    """Compute expected payoff vector for a player."""
    return U @ opponent_strategy if axis == 1 else opponent_strategy @ U


from scipy.optimize import fixed_point

def softmax(x, tau):
    """Compute the softmax with temperature tau."""
    x_max = np.max(x)  # for numerical stability
    exp_x = np.exp((x - x_max) / tau)
    return exp_x / np.sum(exp_x)

def qre_fixed_point(U1, U2, tau=1.0, tol=1e-8, max_iter=500):
    """
    Solve for the Quantal Response Equilibrium using logit QRE.
    
    Parameters:
    - U1: Payoff matrix for player 1 (m x n)
    - U2: Payoff matrix for player 2 (m x n)
    - tau: Rationality parameter (lower = more random, higher = more rational)
    
    Returns:
    - p: Mixed strategy of player 1 (length m)
    - q: Mixed strategy of player 2 (length n)
    """
    m, n = U1.shape
    p = np.ones(m) / m  # initialize player 1 strategy
    q = np.ones(n) / n  # initialize player 2 strategy

    for _ in range(max_iter):
        p_prev, q_prev = p.copy(), q.copy()

        # Best response for player 1 given q
        EU1 = U1 @ q  # expected utility of each action for player 1
        p = softmax(EU1, tau)

        # Best response for player 2 given p
        EU2 = U2.T @ p  # expected utility of each action for player 2
        q = softmax(EU2, tau)

        # Check convergence
        if np.linalg.norm(p - p_prev) + np.linalg.norm(q - q_prev) < tol:
            break

    return p, q



def sparsemax(z):
    """
    Sparsemax as defined by Martins & Astudillo (2016)
    """
    N = len(z)
    z = np.asarray(z)
    z_sorted = np.sort(z)[::-1]
    z_cumsum = np.cumsum(z_sorted)
    k_array = np.arange(1, len(z) + 1)
    t_hat = (z_cumsum - 1) / k_array
    support = z_sorted > t_hat
    k_max = support.sum()
    tau = (z_cumsum[k_max - 1] - 1) / k_max
    return np.maximum(z - tau, 0.0)


def seob_strategy(u, marginal='exponential', params=None, step_size=0.001):
    """
    Compute the SE-OB strategy for a given utility vector u.
    
    Args:
        u: 1D array of expected payoffs (shape: [K])
        marginal: 'exponential', 'uniform', or 'tsallis'
        params: dict of parameters, e.g., {'eta': ..., 'gamma': ..., 'q': ...}
        
    Returns:
        p: SE-OB strategy (array of shape [K])
    """

    def proj_simplex(x):
        sorted_x = np.sort(x)[::-1]
        cumsum = np.cumsum(sorted_x)
        rho = np.where(sorted_x > (cumsum - 1) / (np.arange(K) + 1))[0][-1]
        theta = (cumsum[rho] - 1) / (rho + 1)
        return np.maximum(x - theta, 0)

    u = np.array(u)
    K = len(u)
    
    if params is None:
        params = {}

    if marginal == 'exponential':
        # τ_i ∝ η_i * exp(u_i / γ)
        eta = params.get('eta', np.ones(K))
        gamma = params.get('gamma', 1.0)
        logits = eta * np.exp(u / gamma)
        p = logits / logits.sum()

    elif marginal == 'uniform':
        # τ = sparsemax(u / γ)
        gamma = params.get('gamma', 1.0)
        v = (u / gamma + 1)/(2*K)
        p = sparsemax(v)


    elif marginal == 'tsallis':
        # Solve: max_{p in simplex} <p, u> - γ/(q-1) * sum_i η_i [(p_i/η_i)^q - (p_i/η_i)]
        eta = params.get('eta', np.ones(K))
        gamma = 2.0 * params.get('gamma', 1.0)
        q = params.get('q', 2.0)

        # Closed-form solution doesn't exist; use projected gradient ascent
        def objective(p):
            term1 = np.dot(p, u)
            term2 = - gamma * (1/(q - 1)) * np.sum(eta * ((p / eta) ** q - (p / eta)))
            return term1 + term2
        
        def grad(p):
            return u - gamma * q * ((p / eta) ** (q - 1)) / (q - 1) + gamma / (q - 1)


        p = np.ones(K) / K
        for _ in range(100):
            p += 0.01 * grad(p)
            p = proj_simplex(p)

    else:
        raise ValueError(f"Unknown marginal: {marginal}")
    
    return p


def compute_seob_equilibrium(U1, U2, marginal, params, max_iter=100, tol=1e-6):
    N = U1.shape[0]
    # Initialize both players with uniform strategy
    p1 = np.ones(N) / N
    p2 = np.ones(N) / N

    for iteration in range(max_iter):
        u1 = expected_payoff(U1, p2, axis=1)
        new_p1 = seob_strategy(u1, marginal, params)

        u2 = expected_payoff(U2.T, p1, axis=0)
        new_p2 = seob_strategy(u2, marginal, params)

        delta = np.linalg.norm(new_p1 - p1) + np.linalg.norm(new_p2 - p2)
        p1, p2 = new_p1, new_p2

        if delta < tol:
            break

    return p1, p2
