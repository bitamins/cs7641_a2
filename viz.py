import numpy as np 
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import os



def plot_algs(df,title,x='difficulty',y='epoch',label='label',score='score',filename='viz_py.png',logscale=False):
    plt.clf()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    if logscale: plt.yscale('log')

    plt.grid()

    colors = ['r','g','b','#aaaaaa']
    algs = df.groupby(label).count().index
    for i,alg in enumerate(algs):
        mask = df[label] == alg
        tmp = df[mask]
        plt.plot(tmp[x], tmp[y], 'o-', color=colors[i],
                label=alg,alpha=0.8)

    plt.legend(loc="best")
    plt.savefig('plots/' + '_'.join([title,y,filename]))


def plot_nn(df,title,x,y,lines,filename='nn_py.png',logscale=False):
    plt.clf()
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    if logscale: plt.yscale('log')

    plt.grid()

    colors = ['r','g']
    for i,l in enumerate(lines):
        plt.plot(df[x],df[l],'o-',color=colors[i],label=l,alpha=0.8)

    plt.legend(loc="best")
    plt.savefig('plots/nn/' + '_'.join([title,y,filename]))

def plot_all():
    
    
    outFile = 'py.png'
    
    
    
    # fourpeaks_df = pd.read_csv('fourpeaks_difficulty_all.csv')
    
    # filepath = os.path.join('output_data','fourpeaks_difficulty_all.csv')
    # knapsack_df = pd.read_csv(filepath)
    # plot_algs(fourpeaks_df,'FourPeaks_All',x='difficulty',y='epoch',label='label',score='score',filename=outFile)
    # plot_algs(fourpeaks_df,'FourPeaks_All',x='difficulty',y='time',label='label',score='score',filename=outFile)
    # plot_algs(fourpeaks_df,'FourPeaks_All',x='difficulty',y='score',label='label',score='score',filename=outFile)
    
    # flipflop_df = pd.read_csv('flipflop_difficulty_all.csv')
    
    # filepath = os.path.join('output_data','flipflop_difficulty_all.csv')
    # knapsack_df = pd.read_csv(filepath)
    # plot_algs(flipflop_df,'FlipFlop_All',x='difficulty',y='epoch',label='label',score='score',filename=outFile,logscale=True)
    # plot_algs(flipflop_df,'FlipFlop_All',x='difficulty',y='time',label='label',score='score',filename=outFile)
    # plot_algs(flipflop_df,'FlipFlop_All',x='difficulty',y='score',label='label',score='score',filename=outFile)
    
    filepath = os.path.join('output_data','knapsack_difficulty_all.csv')
    knapsack_df = pd.read_csv(filepath)
    plot_algs(knapsack_df,'Knapsack_All',x='difficulty',y='epoch',label='label',score='score',filename=outFile,logscale=True)
    plot_algs(knapsack_df,'Knapsack_All',x='difficulty',y='time',label='label',score='score',filename=outFile)
    plot_algs(knapsack_df,'Knapsack_All',x='difficulty',y='score',label='label',score='score',filename=outFile)
    
def plot_all_nn():
    outFile = 'nn_py.png'
        
    filepath = os.path.join('output_data','nn','wine_nn_backprop.csv')
    relu_df = pd.read_csv(filepath)
    plot_nn(relu_df,'backprop',x='epoch',y='accuracy',lines=['testAcc','trainAcc'],filename=outFile)
    plot_nn(relu_df,'backprop',x='epoch',y='error',lines=['testErr','trainErr'],filename=outFile)
    
    filepath = os.path.join('output_data','nn','wine_nn_RHC.csv')
    rhc_df = pd.read_csv(filepath)
    plot_nn(rhc_df,'RHC',x='epoch',y='accuracy',lines=['testAcc','trainAcc'],filename=outFile)
    plot_nn(rhc_df,'RHC',x='epoch',y='error',lines=['testErr','trainErr'],filename=outFile)
    
    filepath = os.path.join('output_data','nn','wine_nn_SA.csv')
    sa_df = pd.read_csv(filepath)
    plot_nn(sa_df,'SA',x='epoch',y='accuracy',lines=['testAcc','trainAcc'],filename=outFile)
    plot_nn(sa_df,'SA',x='epoch',y='error',lines=['testErr','trainErr'],filename=outFile)
    
    filepath = os.path.join('output_data','nn','wine_nn_GA.csv')
    ga_df = pd.read_csv(filepath)
    plot_nn(ga_df,'GA',x='epoch',y='accuracy',lines=['testAcc','trainAcc'],filename=outFile)
    plot_nn(ga_df,'GA',x='epoch',y='error',lines=['testErr','trainErr'],filename=outFile)
    
def run():
    # plot_all()
    plot_all_nn()



if __name__ == '__main__':
    run() 