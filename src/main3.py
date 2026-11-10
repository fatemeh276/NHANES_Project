"""
Phase 2 Analysis Script - NHANES Project (Path A)
Author: Fatemeh Aminizadeh
Course: Statistics - Fall 1405

This script covers:
1. Marginal, Joint, and Conditional Probabilities (Gender & Age subgroups)
2. Empirical Independence Verification
3. Bayes' Theorem Update
4. Binomial Distribution Modeling & Simulation
5. Continuous Variable Distribution, Normal Approximation, Expected Value & Variance
6. Covariance & Scatter Plot (BMI vs Blood Pressure)
7. Sampling Weights Impact (Unweighted vs Weighted Proportions with WTMEC2YR)
"""

import os
from functools import reduce
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Visual configurations
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

base_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(base_dir, '../figures')
os.makedirs(figures_dir, exist_ok=True)


def load_and_clean_data():
    """Load, merge, and clean raw NHANES files."""
    data_dir = os.path.join(base_dir, '../data')
    try:
        demo = pd.read_sas(os.path.join(data_dir, 'DEMO_L.XPT'))
        bpx = pd.read_sas(os.path.join(data_dir, 'BPXO_L.XPT'))
        bmx = pd.read_sas(os.path.join(data_dir, 'BMX_L.XPT'))
        tchol = pd.read_sas(os.path.join(data_dir, 'TCHOL_L.XPT'))
        smq = pd.read_sas(os.path.join(data_dir, 'SMQ_L.XPT'))
        paq = pd.read_sas(os.path.join(data_dir, 'PAQ_L.XPT'))
        alq = pd.read_sas(os.path.join(data_dir, 'ALQ_L.XPT'))
    except FileNotFoundError:
        print("Error: Raw data files not found in the 'data' directory.")
        return None

    # Merge on SEQN
    dfs = [demo, bpx, bmx, tchol, smq, paq, alq]
    df = reduce(lambda left, right: pd.merge(left, right, on='SEQN', how='inner'), dfs)

    # Filter adults (Age >= 20)
    df = df[df['RIDAGEYR'] >= 20].copy()

    # Data auditing
    df['PAD680'] = df['PAD680'].replace({7777: np.nan, 9999: np.nan})
    df['PAD680'] = np.where((df['PAD680'] < 0) | (df['PAD680'] > 1440), np.nan, df['PAD680'])
    df['BMXWT'] = np.where((df['BMXWT'] <= 0) | (df['BMXWT'] > 250), np.nan, df['BMXWT'])

    if 'LBXTC' in df.columns:
        df['LBXTC'] = np.where(df['LBXTC'] <= 0, np.nan, df['LBXTC'])

    # Clean BP measurements across all readings
    for i in range(1, 4):
        sys_col = f'BPXOSY{i}'
        dia_col = f'BPXODI{i}'
        if sys_col in df.columns and dia_col in df.columns:
            invalid = (df[sys_col] <= 0) | (df[dia_col] <= 0) | (df[dia_col] >= df[sys_col])
            df.loc[invalid, [sys_col, dia_col]] = np.nan

    # Calculate Mean Systolic BP
    df['BPX_SY_MEAN'] = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)

    # Binary Events definition:
    # H: Hypertension (Mean Systolic BP >= 130 mmHg - AHA/ACC Clinical Guideline)
    # A: Obesity (BMI >= 30.0 kg/m^2 - WHO Standard)
    df['Hypertension'] = np.where(df['BPX_SY_MEAN'].isna(), np.nan, (df['BPX_SY_MEAN'] >= 130.0).astype(float))
    df['Obesity'] = np.where(df['BMXBMI'].isna(), np.nan, (df['BMXBMI'] >= 30.0).astype(float))

    return df


def main():
    print("=================================================================")
    print("  Executing Phase 2: Probability, Distributions & Sampling Weights")
    print("=================================================================")

    df = load_and_clean_data()
    if df is None:
        return

    # Data subset with complete records for event analysis
    df_events = df.dropna(subset=['Hypertension', 'Obesity', 'RIAGENDR', 'RIDAGEYR']).copy()
    n_total = len(df_events)
    print(f"\nTotal valid observations for probability analysis: {n_total}")

    # -------------------------------------------------------------------------
    # Tasks 1, 2 & Path A Requirement 1: Marginal, Joint, and Conditional Probs
    # -------------------------------------------------------------------------
    p_H = df_events['Hypertension'].mean()
    p_A = df_events['Obesity'].mean()
    p_H_and_A = ((df_events['Hypertension'] == 1) & (df_events['Obesity'] == 1)).mean()

    # Conditional Probabilities
    p_H_given_A = p_H_and_A / p_A  # P(Hypertension | Obese)
    p_not_A = 1.0 - p_A
    p_H_and_not_A = ((df_events['Hypertension'] == 1) & (df_events['Obesity'] == 0)).mean()
    p_H_given_not_A = p_H_and_not_A / p_not_A  # P(Hypertension | Non-Obese)

    # Subgroups: Gender (Path A Requirement 1)
    men = df_events[df_events['RIAGENDR'] == 1.0]
    women = df_events[df_events['RIAGENDR'] == 2.0]
    p_H_men = men['Hypertension'].mean()
    p_H_women = women['Hypertension'].mean()

    # Subgroups: Age (Path A Requirement 1)
    older = df_events[df_events['RIDAGEYR'] >= 50]
    younger = df_events[df_events['RIDAGEYR'] < 50]
    p_H_older = older['Hypertension'].mean()
    p_H_younger = younger['Hypertension'].mean()

    print("\n-------------------------------------------------------------")
    print("1, 2 & 3. Marginal, Joint, and Conditional Probabilities")
    print("-------------------------------------------------------------")
    print(f"P(Hypertension) [P(H)]: {p_H:.4f}")
    print(f"P(Obesity) [P(A)]: {p_A:.4f}")
    print(f"P(Hypertension ∩ Obesity): {p_H_and_A:.4f}")
    print(f"P(Hypertension | Obese): {p_H_given_A:.4f}")
    print(f"P(Hypertension | Non-Obese): {p_H_given_not_A:.4f}")
    print(f"P(Hypertension | Men): {p_H_men:.4f}")
    print(f"P(Hypertension | Women): {p_H_women:.4f}")
    print(f"P(Hypertension | Age >= 50): {p_H_older:.4f}")
    print(f"P(Hypertension | Age < 50): {p_H_younger:.4f}")

    # -------------------------------------------------------------------------
    # Task 4: Empirical Independence Verification
    # -------------------------------------------------------------------------
    indep_prod = p_H * p_A
    diff_indep = abs(p_H_and_A - indep_prod)
    print("\n-------------------------------------------------------------")
    print("4. Empirical Independence Evaluation (Hypertension vs Obesity)")
    print("-------------------------------------------------------------")
    print(f"Empirical P(H ∩ A): {p_H_and_A:.4f}")
    print(f"Product of marginals P(H) * P(A): {indep_prod:.4f}")
    print(f"Absolute difference: {diff_indep:.4f}")
    if diff_indep > 0.01:
        print("Verdict: P(H ∩ A) != P(H) * P(A). The events are statistically dependent.")
    else:
        print("Verdict: The data are consistent with the hypothesis of independence.")

    # -------------------------------------------------------------------------
    # Task 9: Bayes' Theorem Update
    # -------------------------------------------------------------------------
    # P(Obesity | Hypertension) = [P(Hypertension | Obesity) * P(Obesity)] / P(Hypertension)
    p_bayes_A_given_H = (p_H_given_A * p_A) / p_H
    empirical_A_given_H = ((df_events['Obesity'] == 1) & (df_events['Hypertension'] == 1)).sum() / (df_events['Hypertension'] == 1).sum()

    print("\n-------------------------------------------------------------")
    print("9. Bayes' Theorem Probability Update")
    print("-------------------------------------------------------------")
    print(f"P(Obesity | Hypertension) via Bayes' formula: {p_bayes_A_given_H:.4f}")
    print(f"P(Obesity | Hypertension) direct empirical calculation: {empirical_A_given_H:.4f}")
    print(f"Discrepancy: {abs(p_bayes_A_given_H - empirical_A_given_H):.6f}")

    # -------------------------------------------------------------------------
    # Task 5 & Path A Requirement 2: Binomial Distribution for Counts
    # -------------------------------------------------------------------------
    n_sample = 20
    n_simulations = 1000
    p_success = p_H

    # Monte Carlo simulation of drawing independent sub-samples of size n=20
    np.random.seed(42)
    sample_counts = [df_events['Hypertension'].sample(n=n_sample, replace=False).sum() for _ in range(n_simulations)]

    x_vals = np.arange(0, n_sample + 1)
    binom_pmf = stats.binom.pmf(x_vals, n_sample, p_success)
    counts_series = pd.Series(sample_counts).value_counts(normalize=True).reindex(x_vals, fill_value=0)

    plt.figure(figsize=(9, 5))
    plt.bar(x_vals - 0.15, counts_series.values, width=0.3, label="Empirical Simulation (Sub-samples)", color="#2b5c8f", alpha=0.85)
    plt.bar(x_vals + 0.15, binom_pmf, width=0.3, label=f"Theoretical Binomial B(n={n_sample}, p={p_success:.2f})", color="#e67e22", alpha=0.85)
    plt.title(f"Comparison of Hypertension Count in Samples (n={n_sample}) with Binomial PMF", fontsize=12)
    plt.xlabel("Number of Individuals with Hypertension (k)", fontsize=11)
    plt.ylabel("Relative Frequency / Probability", fontsize=11)
    plt.xticks(x_vals)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '05_Binomial_Comparison.png'), dpi=300)
    plt.close()
    print("\nFigure saved: figures/05_Binomial_Comparison.png")

    # -------------------------------------------------------------------------
    # Tasks 6, 7 & Path A Requirement 3: Continuous Distribution & Normal Fit
    # -------------------------------------------------------------------------
    bp_clean = df['BPX_SY_MEAN'].dropna()
    mean_bp = bp_clean.mean()
    var_bp = bp_clean.var()
    std_bp = bp_clean.std()
    skew_bp = stats.skew(bp_clean)
    kurt_bp = stats.kurtosis(bp_clean)

    print("\n-------------------------------------------------------------")
    print("6 & 7. Expected Value, Variance & Normal Distribution Evaluation")
    print("-------------------------------------------------------------")
    print(f"Expected Value (Mean): {mean_bp:.2f} mmHg")
    print(f"Variance: {var_bp:.2f}")
    print(f"Standard Deviation: {std_bp:.2f} mmHg")
    print(f"Skewness: {skew_bp:.3f} (positive skew / right-tailed)")
    print(f"Excess Kurtosis: {kurt_bp:.3f}")

    # Plot Distribution + Normal Curve and Q-Q Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(bp_clean, bins=35, stat="density", color="#85c1e9", edgecolor="#34495e", ax=axes[0])
    x_axis = np.linspace(bp_clean.min(), bp_clean.max(), 300)
    axes[0].plot(x_axis, stats.norm.pdf(x_axis, mean_bp, std_bp), color="#c0392b", linewidth=2.2, label=f"Normal Fit (μ={mean_bp:.1f}, σ={std_bp:.1f})")
    axes[0].set_title("Empirical Distribution vs. Normal Density", fontsize=12)
    axes[0].set_xlabel("Mean Systolic BP (mmHg)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].legend()

    # Q-Q plot
    stats.probplot(bp_clean, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_markerfacecolor('#2980b9')
    axes[1].get_lines()[0].set_markeredgecolor('#2980b9')
    axes[1].get_lines()[0].set_alpha(0.4)
    axes[1].get_lines()[1].set_color('#c0392b')
    axes[1].get_lines()[1].set_linewidth(2)
    axes[1].set_title("Normal Q-Q Plot for Systolic BP", fontsize=12)
    axes[1].set_xlabel("Theoretical Quantiles", fontsize=11)
    axes[1].set_ylabel("Sample Quantiles", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '06_Normal_Approximation_QQ.png'), dpi=300)
    plt.close()
    print("Figure saved: figures/06_Normal_Approximation_QQ.png")

    # -------------------------------------------------------------------------
    # Task 8: Two Random Variables and Covariance
    # -------------------------------------------------------------------------
    df_cov = df.dropna(subset=['BMXBMI', 'BPX_SY_MEAN'])
    cov_matrix = np.cov(df_cov['BMXBMI'], df_cov['BPX_SY_MEAN'])
    cov_val = cov_matrix[0, 1]

    print("\n-------------------------------------------------------------")
    print("8. Two Random Variables & Covariance Analysis")
    print("-------------------------------------------------------------")
    print(f"Covariance between BMI and Systolic BP: {cov_val:.3f}")
    print("Direction of association: Positive (Higher BMI is associated with higher Systolic BP).")

    plt.figure(figsize=(8, 6))
    sample_for_scatter = df_cov.sample(n=min(2500, len(df_cov)), random_state=42)
    sns.regplot(
        x='BMXBMI',
        y='BPX_SY_MEAN',
        data=sample_for_scatter,
        scatter_kws={'alpha': 0.25, 'color': '#16a085'},
        line_kws={'color': '#d35400', 'linewidth': 2}
    )
    plt.title(f"Scatter Plot & Trend Line: BMI vs Systolic BP (Cov={cov_val:.2f})", fontsize=12)
    plt.xlabel("Body Mass Index (BMI in kg/m^2)", fontsize=11)
    plt.ylabel("Mean Systolic BP (mmHg)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '07_Covariance_Scatter.png'), dpi=300)
    plt.close()
    print("Figure saved: figures/07_Covariance_Scatter.png")

    # -------------------------------------------------------------------------
    # Path A Requirement 4: Sampling Weights Impact on Probability Estimates
    # -------------------------------------------------------------------------
    df_weighted = df.dropna(subset=['Hypertension', 'WTMEC2YR'])
    raw_p_H = df_weighted['Hypertension'].mean()
    weighted_p_H = np.average(df_weighted['Hypertension'], weights=df_weighted['WTMEC2YR'])

    print("\n-------------------------------------------------------------")
    print("Path A Requirement 4: Impact of Survey Weights on Probability")
    print("-------------------------------------------------------------")
    print(f"Unweighted Sample Proportion [P_unweighted(H)]: {raw_p_H * 100:.2f}%")
    print(f"Weighted Population Estimate [P_weighted(H)]: {weighted_p_H * 100:.2f}%")
    print(f"Absolute Difference: {abs(raw_p_H - weighted_p_H) * 100:.2f}%")
    print("=============================================================")
    print("Phase 2 analysis completed successfully.")


if __name__ == "__main__":
    main()