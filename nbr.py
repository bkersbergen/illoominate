import illoominate
import matplotlib.pyplot as plt
import pandas as pd

location='data/tafeng/processed'
train_df = pd.read_csv(f'{location}/train.csv', sep='\t')
validation_df = pd.read_csv(f'{location}/valid.csv', sep='\t')

loo_values = illoominate.data_loo_values(
    train_df=train_df,
    validation_df=validation_df,
    model='tifu',
    metric='ndcg@20',
    params={'m':7, 'k':100, 'r_b': 0.9, 'r_g': 0.7, 'alpha': 0.7, 'seed': 1313},
)
loo_values.to_csv(f'{location}/data_loo_values.csv', index=False)

plt.hist(loo_values['score'], density=False, bins=100)
plt.title('Distribution of Data LOO Values')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Data LOO Values')
plt.savefig(f'{location}/loo.png', dpi=300)
plt.show()

shapley_values = illoominate.data_shapley_values(
    train_df=train_df,
    validation_df=validation_df,
    model='tifu',
    metric='ndcg@20',
    params={'m':7, 'k':100, 'r_b': 0.9, 'r_g': 0.7, 'alpha': 0.7, 'seed': 1313,  'convergence_threshold': .1, 'convergence_check_every': 25},
)

shapley_values.to_csv(f'{location}/data_shapley_values.csv', index=False)
plt.hist(shapley_values['score'], density=False, bins=100)
plt.title('Distribution of Data Shapley Values')
plt.yscale('log')
plt.ylabel('Frequency')
plt.xlabel('Data Shapley Values')
plt.savefig(f'{location}/shapley.png', dpi=300)
plt.show()


negative = shapley_values[shapley_values.score < 0]
corrupt_users = train_df.merge(negative, on='user_id')
