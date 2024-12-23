import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_samples_from_dataframe(
    df, epsilon=0.1, confidence_level=0.95, population_proportion=0.5
):
    """df must contain 'label'"""
    sampled_df = pd.DataFrame()
    predicted_labels = df.label.unique()
    for label in predicted_labels:
        class_df = df[df.label == label]
        class_size = len(class_df)
        sample_size = sample_size_based_on_class_size(
            class_size=class_size,
            epsilon=epsilon,
            confidence_level=confidence_level,
            population_proportion=population_proportion,
        )
        sampled_df = pd.concat(
            [sampled_df, class_df.sample(sample_size, replace=False)]
        )
    return sampled_df


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


def sampling_from_finite_population_loop():
    df = pd.DataFrame(columns=["epsilon", "population_size", "sample_size", "frac"])
    epsilons = [0.2, 0.1, 0.05, 0.02, 0.01]
    populations = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    fractions = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for pop in populations:
        for frac in fractions:
            dataset = [1]*int(pop*(1-frac))
            dataset = dataset + [0]*int(pop*frac)
            np.random.shuffle(dataset)
            for epsilon in epsilons:
                n = sampling_from_a_finite_population(epsilon=epsilon, population_size=pop, pi=frac)
                df = pd.concat([df, pd.DataFrame({"epsilon": epsilon, "population_size": pop, "sample_size": n, "frac":frac}, index=[0])], ignore_index=True)
                # sample = np.random.choice(dataset, size=n, replace=False)
                print(f"epsilon: {epsilon}, pop_size: {pop}, frac: {frac}, sample_size: {n}")
                # score = np.sum(sample)/n
                # print(f"Score: {score}")
                # print(df)
    line = sns.lineplot(data=df, x="population_size", y="sample_size", hue="epsilon")
    # line.set(yscale="log")
    line.set(xscale="log")
    plt.show()
    
    
def sample_strategy_comparison():
    df = pd.DataFrame(columns=["method", "epsilon", "population_size", "sample_size"])
    epsilons = [0.2, 0.1, 0.05, 0.02, 0.01]
    populations = [100, 1000, 10000, 100000] #100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000
    fractions = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    methods = ["fixed", "fraction", "acceptance sampling"]
    for pop in populations:
        for method in methods:
            if method == "fixed":
                n = 50
            elif method == "fraction":
                n = np.ceil(pop*0.1)
            elif method == "acceptance sampling":
                n = sampling_from_a_finite_population(epsilon=0.05, population_size=pop, pi=0.9)
            epsilon = epsilon_from_sampling_size(n, pi=0.9, N=pop)
            df = pd.concat([df, pd.DataFrame({"method": method, "epsilon": epsilon, "population_size": pop, "sample_size": n}, index=[0])], ignore_index=True)
            print(f"epsilon: {epsilon}, pop_size: {pop}, sample_size: {n}")

    plt.figure()
    line = sns.lineplot(data=df, x="population_size", y="sample_size", hue="method") #size="epsilon",
    line.set(yscale="log")
    line.set(xscale="log")
    scatter = sns.scatterplot(data=df, x="population_size", y="sample_size", hue="method",size="epsilon", sizes=(100,1000), marker="o", legend=None) #size="epsilon",
    scatter.set(yscale="log")
    scatter.set(xscale="log")
    plt.show()
    

# def main():
    # sample_size_based_on_class_size(
    #     class_size=10000,
    #     epsilon=0.05,
    #     confidence_level=0.95,
    #     population_proportion=0.5,
    #     verbose=True)

    # df = pd.read_csv("articles_df.csv")
    # df = df.sample(10000)
    # print(df)
    # samples_df = create_samples_from_dataframe(df, confidence_level=0.95, epsilon=0.1)
    # print(samples_df)
    # print(samples_df.label.value_counts())
    # samples_df.to_pickle("samples_df.pkl")
    # sampling_from_finite_population_loop()
    # print(sampling_from_a_finite_population(0.01, 10000, 0.9))
    
    # sample_strategy_comparison()


# if __name__ == "__main__":
#     main()
