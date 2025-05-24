#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import illoominate


# In[35]:


# Define model and parameters
model = "vmis"
metric = "mrr@20"
params = {"m": 200, "k": 200}

location='../data/11306'


train_df = pd.read_csv(f"{location}/train.csv", sep='\t').rename(
    columns={'SessionId':'session_id','ItemId':'item_id','Time':'timestamp'}
)
val_df = pd.read_csv(f"{location}/valid.csv", sep='\t').rename(
    columns={'SessionId':'session_id','ItemId':'item_id','Time':'timestamp'}
)
test_df = pd.read_csv(f"{location}/test.csv", sep='\t').rename(
    columns={'SessionId':'session_id','ItemId':'item_id','Time':'timestamp'}
)

#shapley_df = pd.read_csv(f"{location}/__removal_impact_shapley_importance_mrr@21_eval_mrr@21.csv", names=['seed', 'session_id', 'shapley'])
#shapley_df = shapley_df.groupby('session_id')['shapley'].mean().reset_index()


# In[36]:


train_df


# In[37]:


l = illoominate.train_and_evaluate_for_sbr(train_df, test_df, model, metric, params)


# In[38]:


l


# In[39]:


# Compute session lengths
train_session_length = train_df.groupby("session_id")["item_id"].count()
val_session_length = val_df.groupby("session_id")["item_id"].count()
test_session_length = test_df.groupby("session_id")["item_id"].count()


quantiles = [0, 10, 20, 30, 40, 50, 60, 70, 90, 95, 99, 99.5, 100]

# Compute quantiles for each dataset
train_quantiles = np.percentile(train_session_length, quantiles)
val_quantiles = np.percentile(val_session_length, quantiles)
test_quantiles = np.percentile(test_session_length, quantiles)

quantile_df = pd.DataFrame({
    "Quantile (%)": quantiles,
    "Train": train_quantiles,
    "Validation": val_quantiles,
    "Test": test_quantiles
})

# Plot quantile distributions
plt.figure(figsize=(10, 5))
plt.plot(quantiles, train_quantiles, marker='o', linestyle='-', label="Train", color='blue')
plt.plot(quantiles, val_quantiles, marker='s', linestyle='-', label="Validation", color='green')
plt.plot(quantiles, test_quantiles, marker='^', linestyle='-', label="Test", color='red')

plt.xlabel("Quantile (%)")
plt.ylabel("Session Length")
plt.title("Session Length Quantile Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('session_length_vs_mrr.pdf', dpi=300)
plt.show()


# In[48]:


illoominate.train_and_evaluate_for_sbr(train_df, val_df, model, metric, params)


# In[49]:


# Compute session lengths
train_session_length = train_df.groupby("session_id")["item_id"].count()

# Define length thresholds (percentiles)
percentiles = quantiles
length_thresholds = {p: np.percentile(train_session_length, p) for p in percentiles}

def do_eval(input_df):
    scores = illoominate.train_and_evaluate_for_sbr(input_df, val_df, model, metric, params)
    result = scores['score'][0]
    return result

# Experiment results storage
results = []

for p, threshold in length_thresholds.items():
    filtered_train = train_df[train_df["session_id"].isin(train_session_length[train_session_length <= threshold].index)]
    result = do_eval(filtered_train)
    results.append({"experiment": f"lengthy_sessions_{p}p", "threshold": threshold, "MRR@20": result})
    print(f"Removed sessions >{threshold} items (p{p}), MRR@20: {result}")


# In[50]:


results


# In[51]:


# Extract values for plotting
thresholds = [r["threshold"] for r in results]
mrr_scores = [r["MRR@20"] for r in results]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(thresholds, mrr_scores, marker='o')
plt.title("Impact of Session Length Threshold on MRR@20")
plt.xlabel("Max Session Length (items)")
plt.ylabel("MRR@20")
plt.xscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig('session_length_vs_mrr.pdf', dpi=300)
plt.show()


# In[ ]:


# Compute Data Shapley values
shapley_values = illoominate.data_shapley_values(
    train_df=train_df,
    validation_df=val_df,
    model='vmis',  # Model to be used (e.g., 'vmis' for VMIS-kNN)
    metric='mrr@20',  # Evaluation metric (e.g., Mean Reciprocal Rank at 20)
    params={'m':200, 'k':200, 'seed': 42},  # Model-specific parameters
)


# In[ ]:


shapley_values.to_csv('shapley_values.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




