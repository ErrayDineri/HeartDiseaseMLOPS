"""Exécute une EDA rapide: résumé texte + graphiques standards."""

import sys
import subprocess

def ensure(pkg):
    """Installe un package à la volée s'il est manquant."""
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

ensure('pandas')
ensure('matplotlib')
ensure('seaborn')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    """Génère les artefacts EDA utilisés dans la documentation du projet."""
    df = pd.read_csv('data.csv')

    out_lines = []
    out_lines.append(f"Shape: {df.shape}")
    out_lines.append(', '.join(df.columns.tolist()))
    out_lines.append('\n\nHead:')
    out_lines.append(df.head().to_string())
    out_lines.append('\n\nDtypes:')
    out_lines.append(df.dtypes.to_string())
    out_lines.append('\n\nMissing values:')
    out_lines.append(df.isnull().sum().to_string())
    out_lines.append('\n\nDescribe:')
    out_lines.append(df.describe(include='all').to_string())

    # Si la cible existe, inclure l'équilibre des classes.
    if 'target' in df.columns:
        out_lines.append('\n\nTarget distribution:')
        out_lines.append(df['target'].value_counts().to_string())

    summary_text = '\n'.join(out_lines)
    with open('eda_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(summary_text)

    # Génère des visualisations numériques standard.
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        # Histogrammes pour chaque variable numérique.
        df[numeric].hist(bins=15, figsize=(12, 8))
        plt.tight_layout()
        plt.savefig('histograms.png')
        plt.clf()

        # Carte de corrélation pour repérer les variables redondantes.
        corr = df[numeric].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation matrix')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.clf()

        # Pairplot sur un sous-ensemble pour limiter le temps d'exécution.
        try:
            small = numeric[:6]
            sns.pairplot(df[small].dropna())
            plt.savefig('pairplot.png')
            plt.clf()
        except Exception:
            pass

    # Countplots pour les colonnes non numériques de faible cardinalité.
    categorical = [c for c in df.columns if df[c].nunique() <= 10 and c not in numeric]
    for c in categorical:
        plt.figure(figsize=(6,4))
        sns.countplot(x=c, data=df)
        plt.title(f'Count of {c}')
        plt.tight_layout()
        safe_name = c.replace('/', '_')
        plt.savefig(f'count_{safe_name}.png')
        plt.clf()

if __name__ == '__main__':
    main()
