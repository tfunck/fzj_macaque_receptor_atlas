import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LinearRegression

def get_uncorrelated_receptors(weights, df, threshold:float=5):
    valid_receptors = [ weights['Receptor'].iloc[0] ]

    for i in range(2, weights.shape[0]):
        current_receptor = weights['Receptor'].iloc[i-1]

        test_receptors = valid_receptors  + [current_receptor]

        X = df[test_receptors]

        vif = variance_inflation_factor(X.values, 0)

        print(f'VIF for {test_receptors}: {vif}')
        if float(vif) < threshold:
            valid_receptors.append(current_receptor)

    df = df[valid_receptors]
    return df

def reg_analysis(X:pd.DataFrame, Y:pd.DataFrame,output_dir:str,clobber:bool=False):
    os.makedirs(output_dir, exist_ok=True)

    #ligand_receptor_dict={'ampa':'AMPA', 'kain':'Kainate', 'mk80':'NMDA', 'ly34':'mGluR2/3', 'flum':'GABA$_A$ Benz.', 'cgp5':'GABA$_B$', 'musc':'GABA$_A$ Agonist', 'sr95':'GABA$_A$ Antagonist', 'pire':r'Muscarinic M$_1$', 'afdx':r'Muscarinic M$_2$ (antagonist)','damp':r'Muscarinic M$_3$','epib':r'Nicotinic $\alpha_4\beta_2$','oxot':r'Muscarinic M$_2$ (oxot)', 'praz':r'$\alpha_1$','uk14':r'$\alpha_2$ (agonist)','rx82':r'$\alpha_2$ (antagonist)', 'dpat':r'5-HT$_{1A}$','keta':r'5HT$_2$', 'sch2':r"D$_1$", 'dpmg':'Adenosine 1', 'cellbody':'Cell Body', 'myelin':'Myelin'}
    vif = variance_inflation_factor(X.values, 0)
    print('ViF:',vif)

    # Iterate over Y columns
    for y_col in Y.columns:
         
        y = Y[y_col]
        #y = (y - y.mean()) / y.std()

        # Pair plot between y and each X column
        m = np.sqrt(X.shape[1]).astype(int)
        n = np.ceil(X.shape[1] / m).astype(int)
        plt.close(); plt.clf(); plt.cla()
        plt.figure(figsize=(m*4, n*4))
        for i, x_col in enumerate(X.columns):
            plt.title(x_col)
            plt.subplot(m,n, i+1)
            plt.scatter(X[x_col], y)
        plt.savefig(f'{output_dir}/{y_col}_pairplot.png')
        plt.close(); plt.clf(); plt.cla()

        #Perform multiple linear regression
        reg = LinearRegression()
        
        #reg = ElasticNetCV(cv=5, l1_ratio=[1], alphas=[0.01,0.05,0.1], max_iter=10000) 
        #from sklearn import svm
        #reg = svm.SVR(kernel="linear")
        reg.fit(X, y)

        import statsmodels.api as sm
        reg = sm.OLS(y,X)
        fit = reg.fit()
        print(fit.summary())

        plt.close(); plt.clf(); plt.cla()   
        plt.figure(figsize=(12, 6))
        sns.barplot(x=X.columns, y=fit.params)
        # add stars for significance based on p-values in <fit>
        for i, p in enumerate(fit.pvalues):
            star=''
            if p < 0.05:
                star='*'
            if p < 0.01:
                star='**'
            if p < 0.001:
                star='***'
            
            plt.text(i, max(0, fit.params[i]+0.1), star, fontsize=12, ha='center')
                
        plt.ylabel('Coefficient')

        plt.xticks(rotation=90)
        plt.title(f'{y_col} Coefficients')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{y_col}_coefficients.png')
        plt.close(); plt.clf(); plt.cla()


        #Print results 
        #print(f"Regression results for {y_col}:")
        #print(f"R^2: {reg.score(X, y)}")
        #print(f'Coefficients: {reg.coef_}')
        try :
            print(f'L1 Ratio: {reg.l1_ratio_}')
            print(f'Alpha: {reg.alpha_}')
        except AttributeError:
            pass

        try :
            print(f'Converged after {reg.n_iter_} iterations')
        except AttributeError:
            pass

