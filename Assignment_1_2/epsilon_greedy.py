import numpy as np
import matplotlib.pyplot as plt
from numba import njit # Import the just-in-time compiler to speed up the code

@njit
def epsilon_greedy(K, T, mu, epsilon=0, method=0):
    """
    Epsilon_greedy algorithm for K-armed bandit problem.
    Args:
        K: Number of arms.
        T: Number of time steps.
        mu: Mean rewards for each arm.
        epsilon: Exploration fraction. Default is 0.
        method: Method to compute N. Options: 'linear', 'sublinear'.
            If 0, use the given epsilon (or the default epsilon), i.e. expect linear regrets.
            If 1, epsilon = 1/t for each t, i.e. expect sublinear regrets.
    Returns:
        expected_regret: Expected regret of the algorithm.
    """
    mu_star = np.max(mu) # Optimal arm reward
    
    # We are assuming that the each arm was played once before the start of the algorithm.
    sums = np.zeros(K) # Sum of rewards for each arm
    counts = np.zeros(K) # Number of times each arm was played
    regret = 0 # Regret for each arm

    # Warm-up: play each arm once to get an initial estimate.
    for arm in range(K):
        # Bernoulli reward with probability mu[arm]
        reward = np.random.binomial(1, mu[arm]) # Get reward ~ Bernoulli(mu[arm])
        sums[arm] += reward
        counts[arm] += 1
        regret += mu_star - mu[arm]

    # Start from K because we have already played each arm once
    for t in range(K,T):

        # In case of sublinear method, update epsilon for each time step.
        if method == 1:
            epsilon = 1 / (t + 1)
            
        # Exploration: choose an arm uniformly at random.
        if np.random.rand() < epsilon:
            arm = np.random.randint(K) # Play all arms uniformly

        # Exploitation: compute sample means and choose the best.
        else:
            # Calculate the sample mean reward for each arm so far.
            mu_sample = np.divide(sums, counts)
            arm = np.argmax(mu_sample) # Choose the best arm so far based on sample mean

        # Play the chosen arm and update the sample mean reward and regret.
        reward = np.random.binomial(1, mu[arm]) # Get reward ~ Bernoulli(mu[arm])
        sums[arm] += reward
        counts[arm] += 1
        regret += mu_star - mu[arm]

    return regret


def average_regret(K, T, mu, epsilon=0, method=0, num_runs=100):
    """
    Computes the average regret over multiple simulation runs.
    
    Args:
        K: Number of arms.
        T: Total time steps.
        mu: Mean rewards for each arm.
        epsilon: Exploration fraction. Default is 0.
        method: Method to compute N. Options: 'linear', 'sublinear'.
            If 0, use the given epsilon (or the default epsilon), i.e. expect linear regrets.
            If 1, epsilon = 1/t for each t, i.e. expect sublinear regrets.
        num_runs: Number of simulation runs.
    
    Returns:
        avg_regret: The average regret over the simulation runs.
    """
    regrets = [epsilon_greedy(K, T, mu, epsilon, method) for _ in range(num_runs)]

    return np.mean(regrets)