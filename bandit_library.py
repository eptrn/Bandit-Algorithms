import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from math import floor
import seaborn as sns
import yfinance as yf


class Data():
    def __init__(self,name):
        self.name = name
        print("Asset : {}".format(self.name))

    def get_data(self,start_date,end_date,interval):

        self.start_date=start_date
        self.end_date=end_date
        self.df = yf.download(self.name, start=self.start_date, end=self.end_date,interval = interval)
        self.df.drop(["Adj Close"],axis=1,inplace=True)
        self.R = self.df['Close'].rolling(2).apply(lambda x: x[1]/x[0])      #RETURN IS = CLOSE(day)/CLOSE(day-timedelta(day=1))
        self.R = self.R.dropna()                                        
        print("start date   : {}".format(self.start_date))
        print("start date   : {}".format(self.end_date))
        return pd.DataFrame(self.R)

        
def get_cov(hist):
    if isinstance(hist,pd.DataFrame):
        return(hist.cov()) ##normalized covariance by N-ddof
    else:
        print('Please Input a Pandas Dataframe')
        return(0)
    
def calculate_return_and_cov(hist,k,tau,interval,trade_start_time):
    current_time_k = trade_start_time + datetime.timedelta(days=k-1)
    if interval == '1wk':
        current_time_k = trade_start_time + datetime.timedelta(days=7*k)
    hist_temp = hist[(hist.index > current_time_k - datetime.timedelta(days=tau))&(hist.index < current_time_k)] #start at k-tau and end at k-1 (hist start time is shifted by -tau compared to the visible index)
    E_Rk = hist_temp.mean() 
    Sigma_k = get_cov(hist_temp)
    return(E_Rk,Sigma_k)

def eigen_decomposition(cov_matrix):
    L_temp, H_temp = np.linalg.eigh(cov_matrix)
    H = H_temp[:,::-1]
    L = L_temp[::-1]  # sort the eigenvalues in decreasing order as mentioned in the paper
    return np.diag(np.abs(L)), H

def normalization(n_assets,L,H):
    LP = [l/np.linalg.norm(H[:,n].dot(np.ones(n_assets)))**2 for n,l in enumerate(L)]
    temp = np.diag(LP)
    temp_s = np.array((list(np.sort(temp))))
    LP = np.diag(temp_s)
    for i in range(n_assets):
        H[:,i]=H[:,i]/(np.transpose(H[:,i]).dot(np.ones(n_assets)))
    return LP, H

def get_sharpe_ratios(n_assets,H,Lambdas,E_R):
    portfolio_returns = np.matmul(np.transpose(H),np.array(E_R))
    return(portfolio_returns/np.sqrt(np.abs(Lambdas)))


def objective_function(n_assets,SR,k,tau,choices):
    obj = []
    for i in range(len(SR)):
        obj.append(SR[i]+np.sqrt((2*np.log(k+tau))/(tau+choices[i])))
    return(obj)

def ucb_policy(n_assets,SR,k,tau,choices):
    return(np.argmax(objective_function(n_assets,SR,k,tau,choices)))

# def James_Stein_Estimator(avg_return_estimate): #used to estimate the average return more precisely 
#     alpha = (1/n_assets)
    
#     return(0)

        
# class OrthogonalPortfolio():
#     def __init__(self,n,m,delta_t,l,R_k,tau):
#         self.n = n
#         self.m = m
#         self.delta_t = delta_t
#         self.l = l
#         self.R_k = R_k
#         self.tau = tau
#         return(0)


def factor_decomposition(Lambda_norm,H_norm,SR,l):
    # l is our cutoff 
    Lambda_inf = Lambda_norm[:l]
    Lambda_sup = Lambda_norm[l:]
    H_inf = H_norm[:,:l]
    H_sup = H_norm[:,l:]
    SR_inf = SR[:l]
    SR_sup = SR[l:]
    return(SR_inf,SR_sup,Lambda_inf,Lambda_sup,H_inf,H_sup)

            

        