import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sampling_from_a_finite_population(epsilon=0.1, population_size=1000, pi=0.5):
    z_score = 1.96
    max_variance = (epsilon/z_score)**2
    sample_size = (population_size*(pi*(1-pi))) / ((population_size-1)*max_variance + (pi*(1-pi)))
    sample_size = int(np.round(sample_size))
    return sample_size

def epsilon_from_sampling_size(n, pi=0.5, N=10000):
    z_score = 1.96
    sigma_square = (pi*(1-pi))
    epsilon = z_score*np.sqrt((N*sigma_square/(n*(N-1))) - (sigma_square/(N-1)))
    return epsilon

def sample_size_based_on_class_size(
    class_size,
    epsilon=0.1,
    confidence_level=0.95,
    population_proportion=0.5,
    verbose=False,
):
    z_score = float(norm.ppf(confidence_level + (1 - confidence_level) / 2))
    sample_size_unlimited = (
        z_score**2
        * population_proportion
        * (1 - population_proportion)
        / epsilon**2
    )
    sample_size_limited = int(
        sample_size_unlimited / (1 + (sample_size_unlimited - 1) / class_size)
    )
    if verbose:
        print("Confidence level = %f => z-score = %f" % (confidence_level, z_score))
        print(
            "Sample size for estimated positive population proportion %f should be %i"
            % (population_proportion, sample_size_limited)
        )

    return sample_size_limited
