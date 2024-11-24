# Illoominate - Data Importance for Recommender Systems

Illoominate is a scalable library designed to compute data importance scores for interaction data in recommender systems. It supports the computation of Data Shapley values (DSV) and leave-one-out (LOO) errors, offering insights into the relevance and quality of data in large-scale recommendation models. This library is tailored for sequential kNN based algorithms including session-based recommendation and next-basket recommendation tasks, and it efficiently handles real-world datasets with millions of interactions.


# Illoominate Framework
This repository contains the code for the illoominate framework, which accompanies the scientific manuscript which is under review.

## Important Notice
This code is made available **exclusively for peer review purposes**.

- Any use, reproduction, or modification of this code is **prohibited** without explicit written permission from the authors.
- Upon acceptance of the manuscript, the repository will be updated to include a standalone, pip-installable version of the library, along with all code and experiments described in the paper, released under the open-source Apache License 2.0.

## Copyright
© 2024 Barrie Kersbergen. All rights reserved.

## Overview

Illoominate is implemented in Rust with a Python frontend. It is optimized to scale with datasets containing millions of interactions, commonly found in real-world recommender systems. The library includes KNN-based models VMIS-kNN and TIFU-kNN, used for session-based recommendations and next-basket recommendations.

By leveraging the Data Shapley value, Illoominate helps data scientists and engineers:
- Debug potentially corrupted data
- Improve recommendation quality by identifying impactful data points
- Prune training data for sustainable item recommendations

### Example Use Case: Computing Data Shapley Values

To compute Data Shapley values for a specific recommendation model (e.g., VMIS-kNN), use the `illoominate.data_shapley_values()` function. This function calculates the impact of each data point in your training data based on a specified recommendation model and evaluation metric.

```python
import illoominate
import matplotlib.pyplot as plt
import pandas as pd

# Load training and validation datasets
train_df = pd.read_csv("data/largersample/train.csv", sep='\t')
validation_df = pd.read_csv("data/largersample/valid.csv", sep='\t')

# Compute Data Shapley values
shapley_values = illoominate.data_shapley_values(
    train_df=train_df,
    validation_df=validation_df,
    model='vmis',  # Model to be used (e.g., 'vmis' for VMIS-kNN)
    metric='mrr@20',  # Evaluation metric (e.g., Mean Reciprocal Rank at 20)
    params={'m':100, 'k':100, 'seed': 42},  # Model-specific parameters
)

negative = shapley_values[shapley_values.score < 0]
corrupt_sessions = train_df.merge(negative, on='session_id')

# Visualize the distribution of Data Shapley values
plt.hist(shapley_values['score'], density=False, bins=100)
plt.title('Distribution of Data Shapley Values')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Data Shapley Values')
plt.savefig('images/shapley.png', dpi=300)
plt.show()
```
### Sample Output
The distribution of Data Shapley values can be visualized or used for further analysis.
![Distribution of Data Shapley Values](images/shapley.png)

```python
print(corrupt_sessions)

    session_id	item_id	timestamp	score
0	5076	64	1585507853	-2.931978e-05
1	13946	119	1584189394	-2.606203e-05
2	13951	173	1585417176	-6.515507e-06
3	3090	199	1584196605	-2.393995e-05
4	5076	205	1585507872	-2.931978e-05
...	...	...	...	...
956	13951	5860	1585416925	-6.515507e-06
957	447	3786	1584448579	-5.092383e-06
958	7573	14467	1584450303	-7.107826e-07
959	5123	47	1584808576	-4.295939e-07
960	11339	4855	1585391332	-1.579517e-06
961 rows × 4 columns
```



### Key Features

- Scalable: Optimized for large datasets with millions of interactions.
- Efficient Computation: Uses the KMC-Shapley algorithm to speed up the estimation of Data Shapley values, making it suitable for real-world recommender systems.
- Customizable: Supports multiple recommendation models, including VMIS-kNN (session-based) and TIFU-kNN (next-basket), and supports popular metrics such as MRR, NDCG, Recall, F1 etc.
- Visualization: Easily visualize the distribution of Data Shapley values to analyze data quality and identify potential issues.
- Real-World Application: Focuses on practical use cases, including debugging, data pruning, and improving sustainability in recommendations.



### How KMC-Shapley Optimizes DSV Estimation

KMC-Shapley (K-nearest Monte Carlo Shapley) is a custom-tailored, scalable variant of the Data Shapley value computation, designed specifically for sequential KNN-based recommendation systems. 
KMC-Shapley improves the efficiency of Truncated Data Shapley (TMC-Shapley) by leveraging the sparsity and nearest-neighbor characteristics of the data, making it feasible to apply Data Shapley values in large-scale recommender systems.
By only computing the utility change when necessary (i.e., when a neighbor's addition impacts the top-k set), KMC-Shapley skips redundant computations, significantly reducing the time complexity.


## Installation

Requirements:
- Rust >= 1.82
- Python >= 3.10

To get started with **Illoominate**, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/bkersbergen/illoominate.git
cd illoominate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
maturin develop --release

```


### Conduct experiments from paper
The experiments from the paper are available in Rust code.

Prepare a config file for a dataset, describing the model, model parameters and the evaluation metric.
```bash
$ cat config.toml
[model]
name = "vmis"

[hpo]
k = 50
m = 500

[metric]
name="MRR"
length=20
```

The software expects the config file for the experiment in the same directory as the data files.
```bash
DATA_LOCATION=data/convergence_11306 CONFIG_FILENAME=config.toml cargo run --release --bin mc_convergence_experiment
```

