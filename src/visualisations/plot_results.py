import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(daily_returns, figsize=50, filename="Correlation_Matrix.png"):
    '''
    Plot the correlation matrix of asset returns.
    '''
    plt.figure(figsize=(figsize,figsize))
    sns.heatmap(daily_returns.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(filename, bbox_inches='tight')

def plot_weight_distribution(weights, tickers, filename="Portfolio_Weights.png"):
    '''
    Plot portfolio weight distribution.
    '''
    plt.figure(figsize=(15, 10))
    plt.bar(tickers, weights)
    plt.title("Portfolio Weight Distribution")
    plt.ylabel("Weight")
    plt.ylim(0, max(weights)+0.01)
    plt.xticks(rotation=45)
    plt.savefig(filename, bbox_inches='tight')
