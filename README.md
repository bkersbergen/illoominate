# Illoominate - Data Importance for Recommender Systems

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3705328.3748049-blue.svg)](https://doi.org/10.1145/3705328.3748049)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![RecSys 2025](https://img.shields.io/badge/RecSys-2025-green.svg)](https://recsys.acm.org/recsys25/)
[![PyPI](https://img.shields.io/pypi/v/illoominate.svg)](https://pypi.org/project/illoominate/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/illoominate.svg)](https://pypi.org/project/illoominate/)

<span style="font-variant: small-caps;">Illoominate</span> is a library that implements the KMC-Shapley algorithm for computing data importance scores in recommender systems. The KMC-Shapley algorithm leverages the sparsity and nearest-neighbor structure of sequential kNN-based recommendation models to efficiently compute Data Shapley values (DSV) and leave-one-out (LOO) scores. This algorithmic approach enables scalable data debugging and importance analysis for real-world datasets with millions of interactions in session-based and next-basket recommendation tasks.

This repository contains the official code for the <span style="font-variant: small-caps;">Illoominate</span> framework, which accompanies our paper accepted at RecSys 2025. The paper is available with open access at [https://dl.acm.org/doi/10.1145/3705328.3748049](https://dl.acm.org/doi/10.1145/3705328.3748049).

**Citation:**
```bibtex
@inproceedings{kersbergen-2025-scalable,
author = {Kersbergen, Barrie and Sprangers, Olivier and Karla{\v s}, Bojan and de Rijke, Maarten and Schelter, Sebastian},
booktitle = {RecSys 2025: 19th ACM Conference on Recommender Systems},
date-added = {2025-07-03 20:57:23 +0200},
date-modified = {2025-07-03 21:01:57 +0200},
month = {September},
publisher = {ACM},
title = {Scalable Data Debugging for Neighborhood-based Recommendation with Data Shapley Values},
year = {2025}}
```

### Key Features

- **KMC-Shapley Algorithm**: Implements the novel K-Monte Carlo Shapley algorithm that exploits the sparsity and nearest-neighbor structure of kNN-based models to efficiently compute Data Shapley values.
- **Scalable Data Debugging**: Enables data importance analysis on large datasets with millions of interactions through algorithmic optimization rather than just computational efficiency.
- **Multiple Model Support**: Works with sequential kNN-based recommendation models including VMIS-kNN (session-based) and TIFU-kNN (next-basket), supporting popular metrics such as MRR, NDCG, Recall, F1 etc.
- **Practical Applications**: Designed for real-world use cases including data debugging, quality assessment, data pruning, and sustainable recommendation systems.

## Overview

<span style="font-variant: small-caps;">Illoominate</span> implements the KMC-Shapley algorithm, which is specifically designed for sequential kNN-based recommendation models. The algorithm's key insight is that most data points in kNN-based systems only influence a small subset of predictions through their nearest-neighbor relationships. By exploiting this sparsity, KMC-Shapley avoids redundant computations and enables scalable Data Shapley value estimation.

The library provides a Python frontend with Rust backend implementation, supporting KNN-based models VMIS-kNN (session-based) and TIFU-kNN (next-basket) for real-world recommendation scenarios.

By leveraging the KMC-Shapley algorithm, <span style="font-variant: small-caps;">Illoominate</span> enables data scientists and engineers to:
- Debug potentially corrupted data through efficient importance scoring
- Improve recommendation quality by identifying the most impactful data points
- Prune training data for sustainable and efficient recommendation systems

## Getting Started
### Quick Installation

<span style="font-variant: small-caps;">Illoominate</span> is available via [PyPI](https://pypi.org/project/illoominate/). 

```bash
pip install illoominate
```

Ensure **Python >= 3.10** is installed. We provide precompiled binaries for Linux, Windows and macOS.

### Note
It is recommended to install and run <span style="font-variant: small-caps;">Illoominate</span> from a virtual environment.
If you are using a virtual environment, activate it before running the installation command.

```
python -m venv venv       # Create the virtual environment (Linux/macOS/Windows)   
source venv/bin/activate  # Activate the virtualenv (Linux/macOS)  
venv\Scripts\activate     # Activate the virtualenv (Windows)  
```


# Example Use Cases

### Example 1: Data Leave-One-Out values for Next-Basket Recommendations with TIFU-kNN

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



### Example 2: Computing Data Shapley Values for Session-Based Recommendations

<span style="font-variant: small-caps;">Illoominate</span> computes Data Shapley values to assess the contribution of each data point to the recommendation performance. Below is an example using the public _Now Playing 1M_ dataset.

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

# Identify potentially corrupted sessions
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

### Example 3: Data Shapley values for Next-Basket Recommendations with TIFU-kNN

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

### Example 4: Increasing the Sustainability of Recommendations via Data Pruning

<span style="font-variant: small-caps;">Illoominate</span> supports metrics to include a sustainability term that expresses the number of sustainable products in a given recommendation. SustainableMRR@t as `0.8·MRR@t + 0.2· st` . This utility combines the MRR@t with the “sustainability coverage term” `st` , where `s` denotes the number of sustainable items among the `t` recommended items.

The function call remains the same, you only change the metric to `SustainableMRR`, `SustainableNDCG` or `st` (sustainability coverage term) and provide a list of items that are considered sustainable.

```python
import illoominate
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv('data/rsc15_100k/processed/train.csv', sep='\t')
validation_df = pd.read_csv('data/rsc15_100k/processed/valid.csv', sep='\t')
# rsc15 items considered sustainable. (Randomly chosen for this dataset) 
sustainable_df = pd.read_csv('data/rsc15_100k/processed/sustainable.csv', sep='\t')

importance = illoominate.data_loo_values(
    train_df=train_df,
    validation_df=validation_df,
    model='vmis',
    metric='sustainablemrr@20',
    params={'m':500, 'k':100, 'seed': 42},
    sustainable_df=sustainable_df,
)

plt.hist(importance['score'], density=False, bins=100)
plt.title('Distribution of Data Leave-One-Out Values')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Data Leave-One-Out Values')
plt.savefig('data/rsc15_100k/processed/loo_responsiblemrr.png', dpi=300)
plt.show()

# Prune the training data
threshold = importance['score'].quantile(0.05)  # 5th percentile threshold
filtered_importance_values = importance[importance['score'] >= threshold]
train_df_pruned = train_df.merge(filtered_importance_values, on='session_id')
```

![Distribution of Leave-One-Out Values using ResponsibleMRR metric](https://raw.githubusercontent.com/bkersbergen/illoominate/refs/heads/main/data/rsc15_100k/processed/loo_responsiblemrr.png)

This demonstrates the pruned training dataset, where less impactful or irrelevant interactions have been removed to focus on high-quality data points for model training.

```python
print(train_df_pruned)
	session_id	item_id	timestamp	score
0	3	214716935	1.396437e+09	0.000000
1	3	214832672	1.396438e+09	0.000000
2	7	214826835	1.396414e+09	-0.000003
3	7	214826715	1.396414e+09	-0.000003
4	11	214821275	1.396515e+09	0.000040
...	...	...	...	...
47933	31808	214820441	1.396508e+09	0.000000
47934	31812	214662819	1.396365e+09	-0.000002
47935	31812	214836765	1.396365e+09	-0.000002
47936	31812	214836073	1.396365e+09	-0.000002
47937	31812	214662819	1.396365e+09	-0.000002
```



## Evaluating a Dataset Using the Python API

<span style="font-variant: small-caps;">Illoominate</span> allows you to train a kNN-based model and evaluate it directly using Python.

### Example: Training & Evaluating VMIS-kNN on NowPlaying1M
```python
import illoominate
import pandas as pd

# Load training and validation datasets
train_df = pd.read_csv("data/nowplaying1m/train.csv", sep="\t")
validation_df = pd.read_csv("data/nowplaying1m/valid.csv", sep="\t")

# Define model and evaluation parameters
model = 'vmis'
metric = 'mrr@20'
params = {
    'm': 500,      # session memory
    'k': 100,      # number of neighbors
    'seed': 42     # random seed
}

# Run training and evaluation
scores = illoominate.train_and_evaluate_for_sbr(
    train_df=train_df,
    validation_df=validation_df,
    model=model,
    metric=metric,
    params=params
)

print(f"Evaluation score ({metric}):", scores['score'][0])

```

You can also evaluate against a separate test set if needed:

```python
validation_scores = illoominate.train_and_evaluate_for_sbr(
    train_df=train_df,
    validation_df=val_df,
    model='vmis',
    metric='mrr@20',
    params=params
)

test_scores = illoominate.train_and_evaluate_for_sbr(
    train_df=train_df,
    validation_df=test_df,
    model='vmis',
    metric='mrr@20',
    params=params
)
```


### Supported Recommendation models and Metrics

`model` (str): Name of the model to use. Supported values:
- `vmis`: Session-based recommendation [VMIS-kNN](https://dl.acm.org/doi/10.1145/3514221.3517901). 
- `tifu`: Next-basket recommendation [TIFU-kNN](https://dl.acm.org/doi/10.1145/3397271.3401066).

`metric` (str): Evaluation metric to calculate importance. Supported values:
- `mrr@20` Mean Reciprocal Rank
- `ndcg@20` Normalized Discounted Cumulative Gain
- `st@20` Sustainability coverage
- `hitrate@20` HitRate
- `f1@20` F1
- `precision@20` Precision
- `recall@20` Recall
- `sustainablemrr@20` Combines the MRR with a Sustainability coverage term
- `sustainablendcg@20` Combines the NDCG with a Sustainability coverage term

`params` (dict): Model specific parameters

`sustainable_df` (pd.DataFrame):
- This argument is only mandatory for the sustainable related metrics `st`, `sustainablemrr` or `sustainablendcg`



## The KMC-Shapley Algorithm

KMC-Shapley (K Monte Carlo Shapley) is a novel algorithm specifically designed for computing Data Shapley values in kNN-based recommendation systems. The algorithm's core innovation lies in recognizing that most data points in kNN-based models only influence predictions through their nearest-neighbor relationships, creating inherent sparsity in the importance computation.

**Key Algorithmic Insights:**
- **Sparsity Exploitation**: The algorithm identifies that most data points only affect a small subset of predictions, avoiding unnecessary computations
- **Nearest-Neighbor Structure**: Leverages the kNN model's structure to focus importance calculations on relevant neighbor relationships
- **Monte Carlo Optimization**: Uses strategic sampling to estimate Shapley values while maintaining theoretical guarantees

This algorithmic approach enables scalable Data Shapley value computation on datasets with millions of interactions, making it practical for real-world recommendation systems.

### Development Installation

To get started with developing **<span style="font-variant: small-caps;">Illoominate</span>** or conducting the experiments from the paper, follow these steps:

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
The experiments from the paper are implemented in Rust for performance benchmarking. Rust's memory safety and performance characteristics make it well-suited for the computational benchmarks, while the core KMC-Shapley algorithm provides the algorithmic efficiency for scalable Data Shapley value computation.

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
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

© 2024 All rights reserved.

## Notes
For any queries or further support, please refer to our RecSys 2025 paper.
Contributions and discussions are welcome!


# Releasing a new version of Illoominate
Increment the version number in pyproject.toml

Trigger a build using the CI pipeline in Github, via either:
* A push is made to the main branch with a tag matching *-rc* (e.g., v1.0.0-rc1).
* A pull request is made to the main branch.
* A push occurs on a branch that starts with branch-*.

Download the wheels mentioned in the CI job output and place them in a directory.
Navigate to that directory and then
```bash
twine upload dist/* -u __token__ -p pypi-SomeSecretAPIToken123
```
This will upload all files in the `dist/` directory to PyPI. `dist/` is the directory where the wheel files will be located after you unpack the artifact from GitHub Actions.

