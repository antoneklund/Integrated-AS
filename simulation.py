import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sampling import sampling_from_a_finite_population, epsilon_from_sampling_size
from scipy.stats import binom


POPULATIONS = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
N_CODES = [
    "n=95 eps=0.10",
    "n=370 eps=0.05",
    "n=1936 eps=0.02",
    "n=4899 eps=0.01",
    "n=10000 eps=0.00"
]

def integrated_acceptance_sampling(D, C, pi, acceptance_limit, max_iterations):
    '''This is the code for the algorithm presented in the paper. 
        The functions get_unseen_sample_pair and do_human_evaluation() should
        be replaced when integrated.
        
        A real example of the algorithm is found in plot_truncation_simulation().
    '''
    def get_unseen_sample_pair(D, C):
        document = np.random.randint(0,2)
        classification = np.random.randint(0,2)
        return (document, classification)
    
    def do_human_evaluation(document, classification):
        return document == classification
        
    accepted = 0
    iterations = 0
    iterations_min = np.floor(1 - acceptance_limit)
    truncate = False
    while (not truncate) and (iterations<max_iterations):
        iterations += 1
        (document, classification) = get_unseen_sample_pair()
        pair_ok = do_human_evaluation(document, classification)
        if pair_ok:
            accepted += 1
        if iterations > iterations_min:
            accuracy = accepted/iterations
            truncate = limit_included_in_range_possible_values(iterations, accuracy, pi, acceptance_limit)
    if accuracy >= acceptance_limit:
        print("Accept classifier")
    else:
        print("Reject classifier")


def calculate_acceptance_limit_from_sample_size(n, limit_percentage):
    acceptance_limit = int(np.ceil(limit_percentage * n))
    return acceptance_limit


def plot_truncation_simulation(lang="english", iterations=20, truncation=True, acceptance_limit=0.9, pi=0.9):
    if lang=="english":
        positives = 340
        negatives = 30
    elif lang=="swedish":
        positives = 338
        negatives = 32
    elif lang=="danish":
        positives = 322
        negatives = 48
    elif lang=="seventy":
        positives = 259
        negatives = 111
    elif lang=="eighty":
        positives = 296
        negatives = 74
    elif lang=="ninetyfive":
        positives = 352
        negatives = 18
    
    list_combined = []
    for i in range(positives):
        list_combined.append({"id":i,"correct": 1})
    for i in range(negatives):
        list_combined.append({"id":positives + i,"correct": 0})
    combined_as_dict = {item['id']: item for item in list_combined}
    df = pd.DataFrame.from_dict(combined_as_dict, orient='index')
    decisions = []
    plt.figure(figsize=(4,3))
    for i in range(iterations):
        n = sampling_from_a_finite_population(population_size=10000, epsilon=0.05, pi=pi)
        df_shuffled = df.sample(n).reset_index(drop=True)
        df_shuffled['cumulative_sum'] = df_shuffled["correct"].cumsum()/(df_shuffled.index+1)
        df_shuffled['cumulative_reject']= (df_shuffled.index+1)-df_shuffled["correct"].cumsum()
        df_result = df_shuffled.drop(columns=['correct'])
        df_result["n"] = df_result.index
        df_result["Accuracy"] = df_result["cumulative_sum"]

        sns.lineplot(data=df_result, x="n", y="Accuracy", c="gray", zorder=1)
        if truncation:
            # print(f'language: {lang}, iteration: {i}:')
            decision, acc, truncation_index = sampling_until_truncation(df_shuffled, limit=acceptance_limit, max_samples=n)
            last_index = truncation_index - 1
            last_value = acc
            decisions.append(decision)
            if decision == "reject":
                # print(f'REJECTED at {truncation_index}')
                plt.scatter(last_index, last_value, marker=6, color="r", s=100, zorder=2)
            else:
                # print(f'ACCEPT at {truncation_index}')
                plt.scatter(last_index, last_value, marker=7, color="g", s=100, zorder=2)
 
    plt.axhline(y=acceptance_limit, linewidth=1.5)
    plt.savefig(f"simulation_{lang}.png", bbox_inches="tight")
    plt.show()
    print(decisions)
    
    
def sampling_until_truncation(df, limit, max_samples, min_sample=13):
    sample_index = 0
    accepts_found = 0
    not_truncate = True
    while (not_truncate) and (sample_index < max_samples):
        observation = df["correct"].iloc[sample_index]
        if observation == 1:
            accepts_found += 1
        sample_index += 1
        if sample_index > min_sample:
            acc = accepts_found/sample_index
            not_truncate = limit_included_in_range_possible_values(limit=limit, n=sample_index, acc=acc)
    # print(f"Acc={acc}, epsilon={epsilon_from_sampling_size(n=sample_index, pi=0.9, N=10000)}")
    if acc < limit:
        decision = "reject"
    else:
        decision = "accept"
    return decision, acc, sample_index


def acceptance_probability(n, p, c):
    # Function to calculate probability of acceptance
        return sum(binom.pmf(k, n, p) for k in range(int(c) + 1))

def oc_curve(n_values = [34, 136, 10000], acceptance_limit = 0.9, x_scatter = [0.081, 0.086, 0.128], scatter_labels=["ENG", "SWE", "DAN"]):
    x = np.linspace(0, 1, 100)  # Proportion of defective
    y_values = {}
    acceptance_limits = {}
    for n in n_values:
        c = n * (1-acceptance_limit)
        y_values[n] = [acceptance_probability(n, p, c) for p in x]
        acceptance_limits[n] = c

    plt.figure(figsize=(4, 3))
    for n, y in y_values.items():
        sns.lineplot(x=x, y=y, label=f'n={n}', zorder=1)
    y_scatter = [acceptance_probability(n_values[1], p, n_values[1]/10) for p in x_scatter]
    scatter_df = pd.DataFrame({"x": x_scatter, "y": y_scatter})
    sns.scatterplot(data=scatter_df, x="x", y="y", zorder=2)
    labels = scatter_labels
    for i, label in enumerate(labels):
        plt.text(x_scatter[i], scatter_df["y"][i], label, fontsize=9, ha='right', zorder=3)

    plt.xlim(0.0, 0.3)
    plt.xlabel('True Classifier Error')
    plt.ylabel('Probability of Acceptance')
    plt.title('OC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("oc.png", bbox_inches="tight")
    plt.show()


def limit_included_in_range_possible_values(n, acc, pi=0.9, limit=0.9):
    epsilon = epsilon_from_sampling_size(n=n, pi=pi, N=10000)
    if acc-epsilon >= limit or acc+epsilon < limit:
        return False
    else:
        return True
    

def samples_until_decision(max_samples=136, acceptance_limit=0.9, num_simulations=1000):
    def sampling_until_rejection(defect_rate, max_samples=136, acceptance_limit=0.9):
        num_samples = 0
        accepts_found = 0
        not_truncate = True
        while (not_truncate) and (num_samples < max_samples):                
            if np.random.rand() >= defect_rate:
                accepts_found += 1
            num_samples += 1
            if num_samples > np.ceil((1-acceptance_limit)*max_samples):
                acc = accepts_found/num_samples
                not_truncate = limit_included_in_range_possible_values(limit=acceptance_limit, n=num_samples, acc=acc)
        
        return num_samples

    defect_rates = np.linspace(0.01, 0.50, 30)  # Defect rates from 1% to 30%
    data = []
    for defect_rate in defect_rates:
        model_truncation = "Truncation"
        model_without_truncation = "w.o. Truncation"
        samples_needed_list = [sampling_until_rejection(defect_rate, max_samples, acceptance_limit) for _ in range(num_simulations)]
        avg_samples_needed = np.mean(samples_needed_list)
        data.append((model_truncation, defect_rate, avg_samples_needed))
        data.append((model_without_truncation, defect_rate, max_samples))


    df = pd.DataFrame(data, columns=["Method", "Defect Rate", "Average Samples Needed"])
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(4, 3))
    sns.lineplot(data=df, x="Defect Rate", y="Average Samples Needed", hue="Method", marker='o')
    plt.title("Average Number of Samples Needed")
    plt.xlabel("True Classifier Error")
    plt.ylabel(r"Avg. $n$ Until Stopping")
    plt.grid(True)
    plt.savefig("sample_decision.png", bbox_inches="tight")
    plt.show()


def pi_vs_n():
    N = [100, 1000, 10000, 100000, 1000000]
    pi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsilon = 0.05
    df = pd.DataFrame(columns=["N", "n", "pi"])
    for N_value in N:
        for pi_value in pi:
            for N_value in N:
                n_value = sampling_from_a_finite_population(epsilon=epsilon, pi=pi_value, population_size=N_value)
                add_df = pd.DataFrame({"N": [int(N_value)], "n": [n_value], "pi": [pi_value]})
                df = pd.concat([df, add_df], ignore_index=True)
    df.N = df.N.astype(int)
    df.N = df.N.astype("category")
    sns.color_palette("tab10")
    plt.figure(figsize=(4, 3))
    sns.lineplot(data=df, x="pi", y="n", hue="N")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.xlabel(r"$\pi$")
    plt.savefig("pi_vs_n.png", bbox_inches="tight")
    plt.show()
            