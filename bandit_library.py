import pandas as pd
import numpy as np
import datetime
import yfinance as yf
# import matplotlib.pyplot as plt
# import seaborn as sns

def James_Stein(hist_data):
    sigma_hat = np.std(hist_data) #use sigma hat as the unbiased estimator of the standard deviation
    n_assets = hist_data.shape[1]
    js_coeff = (1-(n_assets-2)*sigma_hat/np.sum(hist_data**2))
    js_estimate = js_coeff*hist_data
    return js_estimate

class Data():
    def __init__(self,name):
        self.name = name
        print("Asset : {}".format(self.name))

    def get_data(self,start_date,end_date,interval):
        self.start_date=start_date # retirer et mettre dans init
        self.end_date=end_date
        self.df = yf.download(self.name, start=self.start_date, end=self.end_date,interval = interval)
        self.R = self.df['Close'].rolling(2).apply(lambda x: x.iloc[1]/x.iloc[0])      #RETURN IS = CLOSE(day)/CLOSE(day-timedelta(day=1))
        self.R = self.R.dropna()                                        
        # print("start date : {}".format(self.start_date))
        # print("end date : {}".format(self.end_date))
        return pd.DataFrame(self.R)
    
    def get_data_JS(self,start_date,end_date,interval):
        self.start_date=start_date
        self.end_date=end_date
        self.df = yf.download(self.name, start=self.start_date, end=self.end_date,interval = interval)
        self.df['Close'] = James_Stein(self.df['Close'])
        self.R = self.df['Close'].rolling(2).apply(lambda x: x.iloc[1]/x.iloc[0])      #RETURN IS = CLOSE(day)/CLOSE(day-timedelta(day=1))
        self.R = self.R.dropna()                                        
        return pd.DataFrame(self.R)

def get_cov(hist):
    if isinstance(hist,pd.DataFrame):
        return hist.cov() ##normalized covariance by N-ddof
    else:
        print('Please Input a Pandas Dataframe')
        return 0
    
def calculate_return_and_cov(hist,k,tau,interval,trade_start_time):
    current_time_k = trade_start_time + datetime.timedelta(days=k-1)
    if interval == '1wk':
        current_time_k = trade_start_time + datetime.timedelta(days=7*k)
    hist_temp = hist[(hist.index > current_time_k - datetime.timedelta(days=tau))&(hist.index < current_time_k)] #start at k-tau and end at k-1 (hist start time is shifted by -tau compared to the visible index)
    E_Rk = hist_temp.mean() # here for James stein ? 
    Sigma_k = get_cov(hist_temp)
    return E_Rk,Sigma_k

# write a function called heavy_tailed_returns that takes in a start date and end date and returns a dataframe containing returns for every given date sampled from a heavy-tailed distribution
def heavy_tailed_returns(start_date,end_date,interval):
    time_list = pd.date_range(start=start_date, end=end_date,freq=interval)
    # Generate heavy-tailed returns in a dataframe df with the same structure as the yfinance dataframe but without downloading anything just sampling
    df = pd.DataFrame(
        {'Date': time_list, 
            'Returns': np.random.pareto(1,size=len(time_list))}
        )
    df = df.dropna()
    return df.set_index('Date')

def eigen_decomposition(cov_matrix):
    L_temp, H_temp = np.linalg.eigh(cov_matrix)
    H = H_temp[:,::-1]
    L = L_temp[::-1]  # sort the eigenvalues in decreasing order as mentioned in the paper
    return np.diag(np.abs(L)), H

def normalization(n_assets,L,H):
    LP = [l/np.linalg.norm(H[:,n].dot(np.ones(n_assets)))**2 for n,l in enumerate(L)] #LP is the diagonal matrix of the eigenvalues, the Lambdas
    temp = np.diag(LP)
    temp_s = np.array((np.sort(temp)))
    LP = np.diag(temp_s)
    for i in range(n_assets):
        H[:,i]=H[:,i]/(np.transpose(H[:,i]).dot(np.ones(n_assets)))
    return LP, H

def get_sharpe_ratios(H,Lambdas,E_R):
    #portfolio_returns = np.matmul(np.transpose(H),np.array(E_R))
    portfolio_returns = H.T @ np.array(E_R)
    return portfolio_returns/np.sqrt(np.abs(Lambdas))

def objective_function_UCB(SR,k,tau,choices):
    obj = []
    for i in range(len(SR)):
        obj.append(SR[i]+np.sqrt((2*np.log(k+tau))/(tau+choices[i])))
    return obj

def get_argmax_from_obj_list(objective_list):
    return np.argmax(objective_list)

# def objective_function_TS(SR,k,tau,choices):
#     obj = []
#     for i in range(len(SR)):
#         obj.append(np.random.normal(SR[i],1/(tau+choices[i])))
#     return(obj)

# def get_argmax_from_obj_list_TS(SR,k,tau,choices):
#     return(np.argmax(objective_function_TS(SR,k,tau,choices)))

# def get_argmax_from_obj_list(SR,k,tau,choices):
#     return(np.argmax(objective_function_UCB(SR,k,tau,choices)))

def factor_decomposition(Lambda_norm,H_norm,SR,l):
    # l is our cutoff for decomposing returns
    Lambda_inf = Lambda_norm[:l]
    Lambda_sup = Lambda_norm[l:]
    H_inf = H_norm[:,:l]
    H_sup = H_norm[:,l:]
    SR_inf = SR[:l]
    SR_sup = SR[l:]
    return SR_inf,SR_sup,Lambda_inf,Lambda_sup,H_inf,H_sup

def get_portfolio_weights(H_inf,Lambda_inf,SR_inf):
    return np.matmul(H_inf,np.diag(SR_inf/Lambda_inf))

def run_obp(interval,l,m,tau,hist_start_time,hist_end_time,trade_start_time,tickers):
    weights = []
    returns = []
    returnsEW = []
    CW_OBP_list =[]
    CW_EW_list=[]
    Theta = []
    n_assets = len(tickers)
    choices_inf = l*[0]
    choices_sup = (n_assets-l)*[0]
    data = Data(tickers)
    hist_returns = data.get_data(hist_start_time,hist_end_time,interval)
    
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

def run_obp_JS(interval,l,m,tau,hist_start_time,hist_end_time,trade_start_time,tickers):
    weights = []
    returns = []
    returnsEW = []
    CW_OBP_list =[]
    CW_EW_list=[]
    Theta = []
    n_assets = len(tickers)
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

# Write an OBP class to store the results of the run_obp function but also with a method that runs the obp function
# and stores the results in the class
class OBP():
    def __init__(self,interval,l,m,tau,hist_start_time,hist_end_time,trade_start_time,tickers):
        self.interval = interval
        self.l = l
        self.m = m
        self.tau = tau
        self.hist_start_time = hist_start_time
        self.hist_end_time = hist_end_time
        self.trade_start_time = trade_start_time
        self.tickers = tickers
        self.returns = []
        self.returnsEW = []
        self.CW_OBP = []
        self.CW_EW = []
        self.weights = []
        self.choices_inf = []
        self.choices_sup = []
        #self.run_obp()
        
    def run_obp(self):
        self.returns,self.returnsEW,self.CW_OBP,self.CW_EW,self.weights,self.choices_inf,self.choices_sup = run_obp(self.interval,self.l,self.m,self.tau,self.hist_start_time,self.hist_end_time,self.trade_start_time,self.tickers)
        
    def run_obp_JS(self):
        self.returns,self.returnsEW,self.CW_OBP,self.CW_EW,self.weights,self.choices_inf,self.choices_sup = run_obp_JS(self.interval,self.l,self.m,self.tau,self.hist_start_time,self.hist_end_time,self.trade_start_time,self.tickers)
