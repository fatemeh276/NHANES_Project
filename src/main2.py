"""Main module for NHANES project data analysis and visualization."""

import os
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Chart visual settings
sns.set_theme(style="whitegrid")

# Create folder for saving figures relative to script location
base_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(base_dir, '../figures')
os.makedirs(figures_dir, exist_ok=True)


def main():
    """Load, clean, analyze, and visualize NHANES dataset."""
    print("--- Phase 1: Data Preparation and Exploration ---")

    # 1. Load files using absolute path relative to main.py location
    try:
        demo = pd.read_sas(os.path.join(base_dir, '../data/DEMO_L.XPT'))
        bpx = pd.read_sas(os.path.join(base_dir, '../data/BPXO_L.XPT'))
        bmx = pd.read_sas(os.path.join(base_dir, '../data/BMX_L.XPT'))
        tchol = pd.read_sas(os.path.join(base_dir, '../data/TCHOL_L.XPT'))
        smq = pd.read_sas(os.path.join(base_dir, '../data/SMQ_L.XPT'))
        paq = pd.read_sas(os.path.join(base_dir, '../data/PAQ_L.XPT'))
        alq = pd.read_sas(os.path.join(base_dir, '../data/ALQ_L.XPT'))
    except FileNotFoundError:
        print(
            "Error: Data files not found. Make sure the files are in the data folder."
        )
        return

    # 2. Merge data
    dfs = [demo, bpx, bmx, tchol, smq, paq, alq]
    df = reduce(lambda left, right: pd.merge(left, right, on='SEQN', how='inner'), dfs)
    print(f"Total individuals after initial merge: {len(df)}")

    # 3. Define analytical population (Adults >= 20 years old)
    df = df[df['RIDAGEYR'] >= 20].copy()
    print(f"Total individuals after age filter (>= 20): {len(df)}")

    # 4. Data auditing and cleaning (handling statistical and logical errors)
    df['PAD680'] = df['PAD680'].replace({7777: np.nan, 9999: np.nan})
    df['SMD641'] = df['SMD641'].replace({77: np.nan, 99: np.nan})

    df['BMXWT'] = np.where(
        (df['BMXWT'] <= 0) | (df['BMXWT'] > 250), np.nan, df['BMXWT']
    )
    df['PAD680'] = np.where(
        (df['PAD680'] < 0) | (df['PAD680'] > 1440), np.nan, df['PAD680']
    )

    if 'LBXTC' in df.columns:
        df['LBXTC'] = np.where(df['LBXTC'] <= 0, np.nan, df['LBXTC'])

    if 'BPXOSY1' in df.columns and 'BPXODI1' in df.columns:
        invalid_bp = (
            (df['BPXOSY1'] <= 0)
            | (df['BPXODI1'] <= 0)
            | (df['BPXODI1'] >= df['BPXOSY1'])
        )
        df.loc[invalid_bp, ['BPXOSY1', 'BPXODI1']] = np.nan

    df['BPX_SY_MEAN'] = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)

    # 5. Compare raw and weighted summaries for total cholesterol
    df_clean_chol = df.dropna(subset=['LBXTC', 'WTMEC2YR'])
    raw_mean = df_clean_chol['LBXTC'].mean()
    weighted_mean = np.average(
        df_clean_chol['LBXTC'], weights=df_clean_chol['WTMEC2YR']
    )

    print("\n--- Total Cholesterol (LBXTC) Mean Comparison ---")
    print(f"Raw mean (unweighted): {raw_mean:.2f} mg/dL")
    print(f"Weighted mean (population estimate): {weighted_mean:.2f} mg/dL")

    # 6. Basic descriptive statistics
    print("\n--- Descriptive Statistics for Quantitative Variables ---")
    vars_to_describe = ['BMXBMI', 'BPX_SY_MEAN', 'LBXTC', 'PAD680']
    print(df[vars_to_describe].describe())

    # 7. Generate, save, and show 4 standard plots
    print("\nGenerating and saving plots in the figures folder...")

    plt.figure(figsize=(8, 6))
    sns.histplot(df['BMXBMI'].dropna(), bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Body Mass Index (BMI)')
    plt.xlabel('BMI (kg/m^2)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(figures_dir, '01_BMI_Histogram.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    df_plot = df.dropna(subset=['BPX_SY_MEAN', 'RIAGENDR']).copy()
    df_plot['Gender'] = df_plot['RIAGENDR'].map({1.0: 'Men', 2.0: 'Women'})
    sns.boxplot(x='Gender', y='BPX_SY_MEAN', data=df_plot, palette='Set2')
    plt.title('Systolic Blood Pressure by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Mean Systolic BP (mmHg)')
    plt.savefig(os.path.join(figures_dir, '02_BP_by_Gender_Boxplot.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    smq_counts = (
        df['SMQ040']
        .value_counts()
        .rename(index={1.0: 'Every day', 2.0: 'Some days', 3.0: 'Not at all'})
    )
    sns.barplot(x=smq_counts.index, y=smq_counts.values, palette='viridis')
    plt.title('Current Cigarette Smoking Status')
    plt.xlabel('Smoking Status')
    plt.ylabel('Count of Participants')
    plt.savefig(os.path.join(figures_dir, '03_Smoking_Status_Barchart.png'))
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='BMXBMI', y='BPX_SY_MEAN', data=df, alpha=0.5, color='coral')
    plt.title('Relationship between BMI and Systolic Blood Pressure')
    plt.xlabel('BMI (kg/m^2)')
    plt.ylabel('Mean Systolic BP (mmHg)')
    plt.savefig(os.path.join(figures_dir, '04_BMI_vs_BP_Scatter.png'))
    plt.show()
    plt.close()

    # 9. Initial Descriptive Comparison (Descriptive comparison between groups)
    print("\n--- Phase 9: Initial Descriptive Comparison by Gender ---")
    df_comp = df.dropna(subset=['BPX_SY_MEAN', 'RIAGENDR']).copy()
    df_comp['Gender_Label'] = df_comp['RIAGENDR'].map({1.0: 'Men', 2.0: 'Women'})
    
    descriptive_comp = df_comp.groupby('Gender_Label')['BPX_SY_MEAN'].agg(
        Mean='mean',
        Std='std',
        Median='median',
        Count='count'
    )
    print(descriptive_comp.round(2))
    print(
        "\nNote: This comparison is strictly descriptive. No inferential statistical "
        "tests or causal claims are made at this stage."
    )

    print("Phase 1 execution completed successfully.")


if __name__ == "__main__":
     main()
