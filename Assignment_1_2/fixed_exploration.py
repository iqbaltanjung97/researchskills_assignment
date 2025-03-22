import numpy as np
import matplotlib.pyplot as plt

def fixed_exploration(K, T, mu, N=None, method='linear'):
    """
    Fixed exploration algorithm for K-armed bandit problem.
    Args:
        K: Number of arms.
        T: Number of time steps.
        mu: Mean rewards for each arm.
        N: Number of exploration rounds. If None, will be computed based on method.
        method: Method to compute N. Options: 'linear', 'sublinear'.
            If 'linear', N = 0.1 * T.
            If 'sublinear', N = K * log(T) / Delta^2.
    Returns:
        expected_regret: Expected regret of the algorithm.
    """
    mu_star = np.max(mu) # Optimal arm reward
    
    if N is None and method == 'linear':
        N = int(0.1 * T)

    elif N is None and method == 'sublinear':
        second_best_mu = np.sort(mu)[-2] # Second best arm reward
        Delta = mu_star - second_best_mu # Difference between optimal and second best arm
        N = int(K * np.log(T) / (Delta**2)) # Optimal number of exploration rounds
        if N > T: # Safety check: N should not exceed T.
            N = T

    sums = np.zeros(K) # Sums of rewards for each arm
    counts = np.zeros(K) # Number of times each arm was played
    regret = 0 # Regret for each arm

    # Exploration phase
    for _ in range(N):
        arm = np.random.randint(K) # Play all arms uniformly
        reward = np.random.binomial(1, mu[arm]) # Get reward ~ Bernoulli(mu[arm])
        counts[arm] += 1 # Count the number of times each arm was played

        # Sum up the rewards for the played arm;
        # Will divide this by count later to get the sample mean.
        sums[arm] += reward
        regret += mu_star - mu[arm] # Update regret for the played arm

    # Get the actual sample mean; exclude arms that were not played
    mu_sample = np.divide(sums, counts, out=np.zeros_like(sums), where=(counts > 0))
    best_arm = np.argmax(mu_sample) # Choose the best arm based on sample mean

    # Exploitation phase
    regret += (T - N) * (mu_star - mu[best_arm])

    return regret


def average_regret(K, T, mu, N=None, method='linear', num_runs=100):
    """
    Computes the average regret over multiple simulation runs.
    
    Args:
        K: Number of arms.
        T: Total time steps.
        mu: Mean rewards for each arm.
        N: Number of exploration rounds. If None, will be computed based on method.
        method: Method to compute N. Options: 'linear', 'sublinear'.
            If 'linear', N = 0.1 * T.  
            If 'sublinear', N = K * log(T) / Delta^2.
        num_runs: Number of simulation runs.
    
    Returns:
        avg_regret: The average regret over the simulation runs.
    """
    regrets = [fixed_exploration(K, T, mu, N, method) for _ in range(num_runs)]

    return np.mean(regrets)