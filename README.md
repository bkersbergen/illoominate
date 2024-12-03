# Illoominate - Data Importance for Recommender Systems

Illoominate is a scalable library designed to compute data importance scores for interaction data in recommender systems. It supports the computation of Data Shapley values (DSV) and leave-one-out (LOO) scores, offering insights into the relevance and quality of data in large-scale sequential kNN-based recommendation models. This library is tailored for sequential kNN-based algorithms including session-based recommendation and next-basket recommendation tasks, and it efficiently handles real-world datasets with millions of interactions.


### Key Features

- Scalable: Optimized for large datasets with millions of interactions.
- Efficient Computation: Uses the KMC-Shapley algorithm to speed up the estimation of Data Shapley values, making it suitable for real-world sequential kNN-based recommendation systems.
- Customizable: Supports multiple recommendation models, including VMIS-kNN (session-based) and TIFU-kNN (next-basket), and supports popular metrics such as MRR, NDCG, Recall, F1 etc.
- Real-World Application: Focuses on practical use cases, including debugging, data pruning, and improving sustainability in recommendations.


# Illoominate Framework
This repository contains the code for the illoominate framework, which accompanies the scientific manuscript which is under review.

## Overview

Illoominate is implemented in Rust with a Python frontend. It is optimized to scale with datasets containing millions of interactions, commonly found in real-world recommender systems. The library includes KNN-based models VMIS-kNN and TIFU-kNN, used for session-based recommendations and next-basket recommendations.

By leveraging the Data Shapley value, Illoominate helps data scientists and engineers:
- Debug potentially corrupted data
- Improve recommendation quality by identifying impactful data points
- Prune training data for sustainable item recommendations


## Installation
- Python >= 3.10

`
pip install illoominate
`


# Example Use Cases
### Example 1: Computing Data Shapley Values for Session-Based Recommendations

Illoominate computes Data Shapley values to assess the contribution of each data point to the recommendation performance. Below is an example using the public _Now Playing 1M_ dataset.

```python
import illoominate
import matplotlib.pyplot as plt
import pandas as pd

# Load training and validation datasets
train_df = pd.read_csv("data/nowplaying1m/train.csv", sep='\t')
validation_df = pd.read_csv("data/nowplaying1m/valid.csv", sep='\t')

# Compute Data Shapley values
shapley_values = illoominate.data_shapley_values(
    train_df=train_df,
    validation_df=validation_df,
    model='vmis',  # Model to be used (e.g., 'vmis' for VMIS-kNN)
    metric='mrr@20',  # Evaluation metric (e.g., Mean Reciprocal Rank at 20)
    params={'m':100, 'k':100, 'seed': 42},  # Model-specific parameters
)

# Visualize the distribution of Data Shapley values
plt.hist(shapley_values['score'], density=False, bins=100)
plt.title('Distribution of Data Shapley Values')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Data Shapley Values')
plt.savefig('images/shapley.png', dpi=300)
plt.show()

# Identify postentially corrupted sessions
negative = shapley_values[shapley_values.score < 0]
corrupt_sessions = train_df.merge(negative, on='session_id')


```
### Sample Output
The distribution of Data Shapley values can be visualized or used for further analysis.
![Distribution of Data Shapley Values](https://raw.githubusercontent.com/bkersbergen/illoominate/refs/heads/main/images/nowplaying1m_shapley.png)

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

### Example 2: Data Shapley values for Next-Basket Recommendations with TIFU-kNN

To compute Data Shapley values for next-basket recommendations, use the _Tafeng_ dataset.


```python
# Load training and validation datasets
train_df = pd.read_csv('data/tafeng/processed/train.csv', sep='\t')
validation_df = pd.read_csv('data/tafeng/processed/valid.csv', sep='\t')

# Compute Data Shapley values
shapley_values = illoominate.data_shapley_values(
train_df=train_df,
validation_df=validation_df,
model='vmis',
metric='mrr@20',
params={'m':500, 'k':100, 'seed': 42, 'convergence_threshold': .1},
)


# Visualize the distribution of Data Shapley values
plt.hist(shapley_values['score'], density=False, bins=100)
plt.title('Distribution of Data Shapley Values')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Data Shapley Values')
plt.savefig('images/shapley.png', dpi=300)
plt.show()
```
![Distribution of Data Shapley Values](https://raw.githubusercontent.com/bkersbergen/illoominate/refs/heads/main/data/tafeng/processed/shapley.png)

### Example 3: Data Leave-One-Out values for Next-Basket Recommendations with TIFU-kNN

```python
# Load training and validation datasets
train_df = pd.read_csv('data/tafeng/processed/train.csv', sep='\t')
validation_df = pd.read_csv('data/tafeng/processed/valid.csv', sep='\t')

#  Data Leave-One-Out values
loo_values = illoominate.data_loo_values(
    train_df=train_df,
    validation_df=validation_df,
    model='tifu',
    metric='ndcg@10',
    params={'m':7, 'k':100, 'r_b': 0.9, 'r_g': 0.7, 'alpha': 0.7, 'seed': 42},
)

# Visualize the distribution of Data Leave-One-Out Values
plt.hist(shapley_values['score'], density=False, bins=100)
plt.title('Distribution of Data LOO Values')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Data Leave-One-Out Values')
plt.savefig('images/loo.png', dpi=300)
plt.show()
```
![Data Leave-One-Out values for Next-Basket Recommendations with TIFU-kNN](https://raw.githubusercontent.com/bkersbergen/illoominate/refs/heads/main/data/tafeng/processed/loo.png)


### How KMC-Shapley Optimizes DSV Estimation

KMC-Shapley (K-nearest Monte Carlo Shapley) enhances the efficiency of Data Shapley value computations by leveraging the sparsity and nearest-neighbor structure of the data. It avoids redundant computations by only evaluating utility changes for impactful neighbors, reducing computational overhead and enabling scalability to large datasets.


 
### Development Installation

To get started with developing **Illoominate** or conducting the experiments from the paper, follow these steps:

Requirements:
- Rust >= 1.82
- Python >= 3.10

1. Clone the repository:
```bash
git clone https://github.com/bkersbergen/illoominate.git
cd illoominate
```

2. Create the python wheel by:
```bash
pip install -r requirements.txt
maturin develop --release
```


#### Conduct experiments from paper
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
DATA_LOCATION=data/tafeng/processed CONFIG_FILENAME=config.toml cargo run --release --bin removal_impact
```

## Licensing and Copyright
This code is made available exclusively for peer review purposes.
Upon acceptance of the accompanying manuscript, the repository will be released under the Apache License 2.0.
© 2024 Barrie Kersbergen. All rights reserved.

## Notes
For any queries or further support, please refer to the scientific manuscript under review.
Contributions and discussions are welcome after open-source release.
