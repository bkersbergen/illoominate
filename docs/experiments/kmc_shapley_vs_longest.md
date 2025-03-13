Experiment Description: Heuristics Evaluation using KMC-Shapley on Training Data vs. Removing Longest Sessions

Objective

This experiment aims to compare two data reduction heuristics for training dataset pruning:
1.	Removing the longest sessions.
2.	Removing sessions with the lowest Data Shapley Value (computed using the KMC-Shapley method).

We hypothesize that the Shapley-based removal strategy will preserve more valuable data for model performance than simply removing the longest sessions.

⸻

Baseline Approach: Removing Longest Sessions
1.	Order sessions in the training data by session length (longest to shortest).
2.	Train the model on the full dataset and evaluate performance.
3.	Iteratively remove the top 1% of longest sessions and retrain the model.
4.	Repeat step 3 until 20% of the training data is removed.
5.	Record model performance at each step.

⸻

Alternative Approach: Removing Low-Shapley Sessions
1.	Compute the Data Shapley Value for each session using the KMC-Shapley method.
2.	Train the model on the full dataset and evaluate performance.
3.	Iteratively remove the 1% of sessions with the lowest Shapley value and retrain the model.
4.	Repeat step 3 until 20% of the training data is removed.
5.	Record model performance at each step.

⸻

Evaluation Criteria
* Compare the model’s predictive performance after each pruning step for both heuristics.
* Analyze whether removing low-Shapley sessions retains better model accuracy compared to removing the longest sessions.
* Investigate the impact of data reduction on overfitting and generalization.
