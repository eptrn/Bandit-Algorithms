import datetime
from bandit_library import *
import warnings
warnings.filterwarnings('ignore')

def main():
    """
    This function initializes the parameters and data required for running the bandit algorithm.
    It sets up the tickers, interval, tau, m, l, trade_start_time, hist_start_time, trade_end_time, and hist_end_time.
    It then creates instances of the OBP class and runs the bandit algorithm.
    Finally, it plots the cumulative wealth for each instance of the OBP class.

    Returns:
        None
    """
    # Initialize the parameters and the data
    tickers = ['TTE.PA','SU.PA','AAPL','MSFT','GOOGL','AMZN','AI.PA','SW.PA','OR.PA','SAN.PA']
    interval = '1d'
    tau = 120
    m = 300
    l = 3
    trade_start_time = datetime.datetime(2022,10,15,17,30)
    if interval == '1d':
        hist_start_time = trade_start_time - datetime.timedelta(days=tau)
        trade_end_time = trade_start_time + datetime.timedelta(days=m)
    if interval == '1wk':
        hist_start_time = trade_start_time - datetime.timedelta(days=7*tau)
        trade_end_time = trade_start_time + datetime.timedelta(days=7*m)
    hist_end_time = trade_end_time
    
    obp_opt = OBP(
        interval,l,m,tau,
        hist_start_time,hist_end_time,trade_start_time,tickers,
        heavy_tail=False
        )
    
    obp_opt.get_optimal_l()
    obp_opt.get_optimal_tau()
    obp_opt.run_obp()
    obp_opt.plot_CW()
    obp_opt.plot_returns()
    obp_opt.plot_weights()

    # Uncomment to see examples of less performing portfolios with different values of l and tau
    # obp_l = OBP(
    #     interval,l,m,tau,
    #     hist_start_time,hist_end_time,trade_start_time,tickers,
    #     heavy_tail=False
    #     )
    # obp_tau = OBP(
    #     interval,l,m,tau,
    #     hist_start_time,hist_end_time,trade_start_time,tickers,
    #     heavy_tail=True
    #     )
    # obp_l.get_optimal_l()
    # obp_tau.get_optimal_tau()
    # obp_tau.run_obp()
    # obp_l.run_obp()
    # obp_tau.plot_CW()
    # obp_l.plot_CW()

    # obp_tau.plot_returns()
    # obp_l.plot_returns()

    # obp_tau.plot_weights()
    # obp_l.plot_weights()


if __name__ == '__main__':
    main()

