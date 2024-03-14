import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from bandit_library import *
import warnings
warnings.filterwarnings('ignore')

# Initialize the parameters and the data
tickers = ['AAPL','MSFT','GOOGL','AMZN','AI.PA']
n_assets = len(tickers)
interval = '1d'
tau = 550
m = 180
trade_start_time = datetime.datetime(2019,9,15,17,30)
if interval == '1d':
    hist_start_time = trade_start_time - datetime.timedelta(days=tau)
    trade_end_time = trade_start_time + datetime.timedelta(days=m)
if interval == '1wk':
    hist_start_time = trade_start_time - datetime.timedelta(days=7*tau)
    trade_end_time = trade_start_time + datetime.timedelta(days=7*m)
hist_end_time = trade_end_time

interval = '1d'
data = Data(tickers)
hist_returns = data.get_data_JS(hist_start_time,hist_end_time,interval)

print('hist_returns', hist_returns.head(5))

def run_obp(interval,l,m,tau):
    weights = []
    returns = []
    returnsEW = []
    CW_OBP_list =[]
    CW_EW_list=[]
    Theta = []
    choices_inf = l*[0]
    choices_sup = (n_assets-l)*[0]
    data = Data(tickers)
    hist_returns = data.get_data_JS(hist_start_time,hist_end_time,interval)
    
    for k in range(1,m+1):
        
        #Step 1 : Estimate Average Return and Covariance Matrix of asset returns
        E_R,Sigma = calculate_return_and_cov(hist_returns,k,tau,interval,trade_start_time)

        #Step 2 : Principal Component Decomposition 
        Lambda, H = eigen_decomposition(Sigma)
        Lambdas = np.diag(Lambda)

        #Step 3 : Normalize
        Lambda_norm, H_norm = normalization(n_assets,Lambda,H)
        Lambdas_norm = np.diag(Lambda_norm)
        Sigma_tilde, Lambda_tilde = np.matmul(np.matmul(H_norm,Sigma),np.transpose(H_norm)),np.matmul(np.matmul(H_norm,Sigma),np.transpose(H_norm))

        #Step 4 : Compute the Sharpe Ratio of each arm
        SR = get_sharpe_ratios(H,Lambdas,E_R)
        #SR = get_sharpe_ratios(H_norm,Lambdas_norm,E_R) #H_norm is H_tilde

        #Step 5 : Compute the adjusted reward function of each arm
        SR_inf,SR_sup,Lambda_inf,Lambda_sup,H_inf,H_sup = factor_decomposition(Lambda_norm,H_norm,SR,l)
        adjusted_rewards_inf = objective_function_UCB(SR_inf,k,tau,choices_inf)
        adjusted_rewards_sup = objective_function_UCB(SR_sup,k,tau,choices_sup)

        #Step 6 : Select an arm for each subset 
        arm_inf = get_argmax_from_obj_list(adjusted_rewards_inf)
        arm_sup = get_argmax_from_obj_list(adjusted_rewards_sup)
        choices_inf[arm_inf]+=1 #updating k_i
        choices_sup[arm_sup]+=1

        #Step 7 : Calculate the portfolio weights and return
        Thetak = np.diag(Lambda_inf)[arm_inf]/(np.diag(Lambda_inf)[arm_inf]+np.diag(Lambda_sup[:,l:])[arm_sup])
        Theta.append(Thetak)
        w_k = (1-Thetak)*H_inf[:,arm_inf] + Thetak*H_sup[:,arm_sup]
        
        #add constraint on the weights
        w_k = np.maximum(0,w_k) # no short selling
        w_k = w_k/np.sum(w_k) # weights sum to 1

        weights.append(w_k)
        mu_k  = np.matmul(np.transpose(w_k),E_R)-1
        returns.append(mu_k)
        mu_k_EW = np.sum(E_R/n_assets)-1
        returnsEW.append(mu_k_EW)
        CW_OBP_list.append(np.prod(np.array(returns)+1))
        CW_EW_list.append(np.prod(np.array(returnsEW)+1))
        
    return returns,returnsEW,CW_OBP_list,CW_EW_list,weights,choices_inf,choices_sup