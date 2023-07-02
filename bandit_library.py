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
        self.time_interval=self.end_date-self.start_date
        self.df = yf.download(self.name, start=self.start_date, end=self.end_date,interval = interval)
        self.df.drop(["Adj Close"],axis=1,inplace=True)
        self.R = self.df['Close'].rolling(2).apply(lambda x: x[1]/x[0])      #RETURN IS = CLOSE(day)/CLOSE(day-timedelta(day=1))
        self.R.dropna(inplace=True)                                        
        print("start date   : {}".format(self.start_date))
        print("start date   : {}".format(self.end_date))
        print("time interval: {}".format(self.time_interval))
        return pd.DataFrame(self.R)
        
def get_cov(hist):
    if isinstance(hist,pd.DataFrame):
        return(hist.cov()) ##normalized covariance by N-ddof
    else:
        print('Please Input a Pandas Dataframe')
        return(0)

def eigen_decomposition(cov_matrix):
    L_temp, H_temp = np.linalg.eigh(cov_matrix)
    L = L_temp[::-1]  # sort the eigenvalues in decreasing order as mentioned in the paper
    H = H_temp[:,::-1]
    return L, H

def normalization(n_assets,L,H):
    LP = [l/np.linalg.norm(H[:,n].dot(np.ones(n_assets)))**2 for n,l in enumerate(L)]
    for i in range(n_assets):
        H[:,i]=H[:,i]/(np.transpose(H[:,i]).dot(np.ones(n_assets)))
    return LP, H

        
class OrthogonalPortfolio():
    def __init__(self,n,m,delta_t,l,R_k,tau):
        self.n = n
        self.m = m
        self.delta_t = delta_t
        self.l = l
        self.R_k = R_k
        self.tau = tau
        