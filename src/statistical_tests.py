"""
Statistical tests for model comparison.

This module provides functions to statistically compare the performance of models
and determine which model is best from a statistical perspective.
"""
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd


def compare_models_with_paired_t_test(model1, model2, X, y, model1_name, model2_name, 
                                       cv=5, metric='roc_auc', random_state=42):
    """
    This test determines if the difference in performance between two models
    is statistically significant. The paired t-test is used because the
    same data folds are used for both models.
    """
    # cv=5: standard 5-fold CV provides good balance between statistical power and computation time
    # metric='roc_auc': appropriate for imbalanced classification, measures model's ability to distinguish classes
    # Use StratifiedKFold to preserve class distribution across folds (critical for imbalanced data)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Get cross-validation scores for both models
    scores1 = cross_val_score(model1, X, y, cv=skf, scoring=metric, n_jobs=-1)
    scores2 = cross_val_score(model2, X, y, cv=skf, scoring=metric, n_jobs=-1)
    
    
    mean1 = scores1.mean()
    std1 = scores1.std()
    mean2 = scores2.mean()
    std2 = scores2.std()
    mean_diff = mean1 - mean2
    
    # Paired t-test
    # H0: The two models have the same mean performance 
    # H1: The models have different mean performance 
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Calculate 95% confidence interval for the difference
    n = len(scores1)
    se_diff = np.std(scores1 - scores2) / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n-1)  # two-tailed
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    results = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'model1_mean': mean1,
        'model1_std': std1,
        'model2_mean': mean2,
        'model2_std': std2,
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'scores1': scores1,
        'scores2': scores2
    }
    
    return results


def print_paired_t_test_results(results):
    """
    Print results of the paired t-test comparison to the console for visual tracking.
    """
    print("\n" + "=" * 80)
    print(f"PAIRED T-TEST: {results['model1_name']} vs {results['model2_name']}")
    print("\n" + "=" * 80)
    print(f"Metric: ROC-AUC (Cross-Validation with {len(results['scores1'])} folds)")
    print("-" * 80)
    print(f"{results['model1_name']}:")
    print(f"  Mean: {results['model1_mean']:.4f} ± {results['model1_std']:.4f}")
    print(f"  Individual fold scores: {[f'{s:.4f}' for s in results['scores1']]}")
    print()
    print(f"{results['model2_name']}:")
    print(f"  Mean: {results['model2_mean']:.4f} ± {results['model2_std']:.4f}")
    print(f"  Individual fold scores: {[f'{s:.4f}' for s in results['scores2']]}")
    print()
    print(f"Difference (Model 1 - Model 2): {results['mean_difference']:.4f}")
    print(f"95% Confidence Interval: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
    print()
    print(f"Paired t-test:")
    print(f"  t-statistic: {results['t_statistic']:.4f}")
    if (results['p_value'] < 0.05):
        print(f"  p-value: {results['p_value']:.4f} < 0.05")
    else:
        print(f"  p-value: {results['p_value']:.4f}")
    print()
    
    if (results['p_value'] < 0.05):
        print("Result is statistically significant difference (p < 0.05)")
        if (results['mean_difference'] > 0):
            print(f"  → {results['model1_name']} is significantly better than {results['model2_name']}")
        else:
            print(f"  → {results['model2_name']} is significantly better than {results['model1_name']}")
    else:
        print("Result is not statistically significant difference (p >= 0.05)")
        print(f"  → {results['model1_name']} and {results['model2_name']} have equivalent performance")
    print("=" * 80)


def mcnemar_test(model1, model2, X_test, y_test, model1_name, model2_name):
    """
    McNemar's test is appropriate for comparing two classifiers on the same
    test set. It focuses on the disagreements between the two models and performs
    a statistical test to determine if the difference is significant.
    """
    # Get predictions from both models
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    
    # Build contingency table
    #           Model 2 correct | Model 2 incorrect
    # Model 1 correct    a      |        b
    # Model 1 incorrect   c      |        d
    
    both_correct = ((y_pred1 == y_test) & (y_pred2 == y_test)).sum()
    model1CorrectModel2Wrong = ((y_pred1 == y_test) & (y_pred2 != y_test)).sum()
    model1WrongModel2Correct = ((y_pred1 != y_test) & (y_pred2 == y_test)).sum()
    bothWrong = ((y_pred1 != y_test) & (y_pred2 != y_test)).sum()
    
    # Perform McNemar's test
    # Using continuity correction for better approximation
    # H0: The two models have the same error rate
    # H1: The models have different error rates
    
    # Calculate chi-square statistic with continuity correction
    b = model1CorrectModel2Wrong
    c = model1WrongModel2Correct
    
    if ((b + c) == 0):
        # if No disagreements, the models make identical predictions
        chi2_stat = 0.0
        p_value = 1.0
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    results = {
        'model1_name': model1_name,
        'model2_name': model2_name,
        'both_correct': both_correct,
        'model1_correct_model2_wrong': model1CorrectModel2Wrong,
        'model1_wrong_model2_correct': model1WrongModel2Correct,
        'both_wrong': bothWrong,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    return results


def print_mcnemar_test_results(results):
    """
    Print results of McNemar's test to the console for visual tracking.
    """
    print("\n" + "=" * 80)
    print(f"MCNEMAR'S TEST: {results['model1_name']} vs {results['model2_name']}")
    print("\n" + "=" * 80)
    print("Contingency Table:")
    print("-" * 80)
    print(f"                    {results['model2_name']}")
    print(f"                  Correct  Incorrect")
    print(f"{results['model1_name']} Correct    {results['both_correct']:6d}    {results['model1_correct_model2_wrong']:6d}")
    print(f"          Incorrect   {results['model1_wrong_model2_correct']:6d}    {results['both_wrong']:6d}")
    print()
    print(f"Disagreements:")
    print(f"  {results['model1_name']} correct, {results['model2_name']} wrong: {results['model1_correct_model2_wrong']}")
    print(f"  {results['model1_name']} wrong, {results['model2_name']} correct: {results['model1_wrong_model2_correct']}")
    print()
    print(f"McNemar\"s test:")
    print(f"  Chi-square statistic: {results['chi2_statistic']:.4f}")
    if (results['p_value'] < 0.05):
        print(f"  p-value: {results['p_value']:.4f} < 0.05")
    else:
        print(f"  p-value: {results['p_value']:.4f}")
    print()
    
    if (results['p_value'] < 0.05):
        print("Result: Statistically significant difference (p < 0.05)")
        if (results['model1_correct_model2_wrong'] > results['model1_wrong_model2_correct']):
            print(f"  → {results['model1_name']} makes significantly more correct predictions")
            print(f"     when {results['model2_name']} is wrong")
        else:
            print(f"  → {results['model2_name']} makes significantly more correct predictions")
            print(f"     when {results['model1_name']} is wrong")
    else:
        print("Result: No statistically significant difference (p >= 0.05)")
        print(f"  → {results['model1_name']} and {results['model2_name']} have equivalent error rates")
    print("=" * 80)


def compare_all_models(models_dict, X_train, y_train, X_test, y_test, cv=5, random_state=42):
    """
    Compare all models using both paired t-test and McNemar's test to determine
    which model is statistically best.
    """
    model_names = list(models_dict.keys())
    models = list(models_dict.values())
    
    all_results = {
        'paired_t_tests': {},
        'mcnemar_tests': {}
    }
    
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON OF ALL MODELS")
    print("=" * 80)
    
    # Compare all pairs of models
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1_name = model_names[i]
            model2_name = model_names[j]
            model1 = models[i]
            model2 = models[j]
            
            # Paired t-test with cross-validation
            t_test_results = compare_models_with_paired_t_test(
                model1, model2, X_train, y_train,
                model1_name, model2_name, cv=cv, random_state=random_state
            )
            print_paired_t_test_results(t_test_results)
            all_results['paired_t_tests'][f"{model1_name}_vs_{model2_name}"] = t_test_results
            
            # McNemar's test on test set
            mcnemar_results = mcnemar_test(
                model1, model2, X_test, y_test,
                model1_name, model2_name
            )
            print_mcnemar_test_results(mcnemar_results)
            all_results['mcnemar_tests'][f"{model1_name}_vs_{model2_name}"] = mcnemar_results
    
    return all_results

