import json

notebook = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.11.0"}
 },
 "cells": [
  {"cell_type": "markdown", "id": "1", "metadata": {}, "source": ["# Data Decay Detection — EDA\n", "Exploratory analysis of 92 S&P 500 stocks across 4 market regimes (2000-2024)"]},
  {"cell_type": "code", "execution_count": None, "id": "2", "metadata": {}, "outputs": [], "source": ["import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nwarnings.filterwarnings('ignore')\nplt.style.use('seaborn-v0_8-darkgrid')\nprint('Libraries loaded!')"]},
  {"cell_type": "code", "execution_count": None, "id": "3", "metadata": {}, "outputs": [], "source": ["df = pd.read_parquet('data/raw/sp500_all_stocks.parquet')\ndf['Date'] = pd.to_datetime(df['Date'])\nprint(f'Shape: {df.shape}')\nprint(df['Regime'].value_counts())\ndf.head()"]},
  {"cell_type": "code", "execution_count": None, "id": "4", "metadata": {}, "outputs": [], "source": ["aapl = df[df['Ticker'] == 'AAPL'].copy().sort_values('Date')\nfig, ax = plt.subplots(figsize=(16, 6))\nax.plot(aapl['Date'], aapl['Close'], color='steelblue', linewidth=1.5)\nax.axvspan(pd.Timestamp('2007-10-01'), pd.Timestamp('2009-03-01'), alpha=0.3, color='red', label='2008 Crisis')\nax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'), alpha=0.3, color='orange', label='COVID Crash')\nax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2024-01-01'), alpha=0.3, color='yellow', label='Volatile')\nax.set_title('AAPL Price 2000-2024 with Market Regimes')\nax.legend()\nplt.savefig('notebooks/chart1_regimes.png', dpi=150)\nplt.show()\nprint('Chart 1 saved!')"]},
  {"cell_type": "code", "execution_count": None, "id": "5", "metadata": {}, "outputs": [], "source": ["fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)\nfor ax, regime, color in zip(axes, ['bull', 'bear', 'volatile'], ['green', 'red', 'orange']):\n    data = df[df['Regime'] == regime]['Daily_Return'].dropna()\n    data = data[data.between(-0.2, 0.2)]\n    ax.hist(data, bins=100, color=color, alpha=0.7)\n    ax.axvline(data.mean(), color='black', linestyle='--', label=f'Mean: {data.mean():.4f}')\n    ax.set_title(f'{regime.capitalize()} Market')\n    ax.legend()\nplt.suptitle('Daily Return Distributions by Regime')\nplt.tight_layout()\nplt.savefig('notebooks/chart2_returns.png', dpi=150)\nplt.show()\nprint('Chart 2 saved!')"]},
  {"cell_type": "code", "execution_count": None, "id": "6", "metadata": {}, "outputs": [], "source": ["aapl_vol = df[df['Ticker'] == 'AAPL'].copy().sort_values('Date')\nfig, ax = plt.subplots(figsize=(16, 5))\nax.plot(aapl_vol['Date'], aapl_vol['Volatility_30d'], color='purple', linewidth=1)\nax.axvspan(pd.Timestamp('2007-10-01'), pd.Timestamp('2009-03-01'), alpha=0.2, color='red', label='2008 Crisis')\nax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'), alpha=0.2, color='orange', label='COVID Crash')\nax.set_title('30-Day Rolling Volatility — Drift Spikes at Crash Events')\nax.legend()\nplt.savefig('notebooks/chart3_volatility.png', dpi=150)\nplt.show()\nprint('Chart 3 saved!')"]},
  {"cell_type": "code", "execution_count": None, "id": "7", "metadata": {}, "outputs": [], "source": ["crisis = df[df['Regime'] == 'bear']\nsector_returns = crisis.groupby('Sector')['Daily_Return'].mean().sort_values()\nfig, ax = plt.subplots(figsize=(12, 6))\ncolors = ['red' if x < 0 else 'green' for x in sector_returns.values]\nax.barh(sector_returns.index, sector_returns.values, color=colors)\nax.set_title('Average Daily Return by Sector During Bear Markets')\nplt.tight_layout()\nplt.savefig('notebooks/chart4_sectors.png', dpi=150)\nplt.show()\nprint('Chart 4 saved!')"]},
  {"cell_type": "code", "execution_count": None, "id": "8", "metadata": {}, "outputs": [], "source": ["features = ['Daily_Return', 'Volatility_30d', 'MA_50', 'MA_200', 'Golden_Cross', 'Volume']\ncorr = df[features].corr()\nfig, ax = plt.subplots(figsize=(10, 8))\nsns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)\nax.set_title('Feature Correlation Heatmap')\nplt.tight_layout()\nplt.savefig('notebooks/chart5_correlation.png', dpi=150)\nplt.show()\nprint('Chart 5 saved!')"]}
 ]
}

with open("notebooks/eda.ipynb", "w") as f:
    json.dump(notebook, f)

print("Notebook updated!")

