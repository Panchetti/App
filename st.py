import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats



st.set_page_config(
    page_title="Option Pricing Models",
    layout="wide",
    initial_sidebar_state="expanded")

st.title('Option Pricing Simulator')
st.markdown("---")

class BlackScholes:

    def __init__(self,S,K,t,sigma,r=0.05):
        self.K = K
        self.S = S
        self.t = t/365
        self.sigma = sigma
        self.r = r

    def Black_Scholes_Call_Option(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        C = stats.norm.cdf(d1, 0.0, 1) * self.S - stats.norm.cdf(d2, 0.0, 1) * self.K * np.exp(-self.r * self.t)
        return format(C, '.4f')

    def Black_Scholes_Put_Option(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        P = stats.norm.cdf(-d2, 0.0, 1) * self.K * np.exp(-self.r * self.t) - stats.norm.cdf(-d1, 0.0, 1) * self.S
        return format(P, '.4f')

bsm = BlackScholes(100.00,100.00,365,0.2)
print(bsm.Black_Scholes_Call_Option())
print(bsm.Black_Scholes_Put_Option())

class Monte_Carlo:
    def __init__(self,K,S_0,t,sigma,simulations,r=0.05):
        self.K = K
        self.S_0 = S_0
        self.t = t/365
        self.sigma = sigma
        self.r = r

        self.N = simulations
        self.num_steps = t
        self.dt = self.t/self.num_steps
        self.simulation_results_S=None

    def simulate_prices(self):

        np.random.seed(20)

        S = np.zeros((self.num_steps,self.N))
        S[0,:]=self.S_0



        for t in range(1,self.num_steps):
            Z = np.random.standard_normal(self.N)
            S[t] = S[t-1]*np.exp((self.r - 0.5*self.sigma**2)*self.dt  + (self.sigma*np.sqrt(self.dt))*Z)
        self.simulation_results_S = S

    def simulate_call(self):
        if self.simulation_results_S is None:
            return -1
        else:
            return np.exp(-self.r*self.t)*1/self.N*np.sum(np.maximum(self.simulation_results_S[-1]-self.K,0))

    def simulate_put(self):
        if self.simulation_results_S is None:
            return -1
        else:
            return np.exp(-self.r*self.t)*1/self.N*np.sum(np.maximum(self.K-self.simulation_results_S[-1],0))

    def plot_simulation_results(self, num_of_movements):
        """Plots specified number of simulated price movements."""
        plt.figure(figsize=(12, 8))
        plt.plot(self.simulation_results_S[:, 0:num_of_movements])
        plt.axhline(self.K, c='k', xmin=0, xmax=self.num_steps, label='Strike Price')
        plt.xlim([0, self.num_steps])
        plt.ylabel('Simulated price movements')
        plt.xlabel('Days in future')
        plt.title(f'First {num_of_movements}/{self.N} Random Price Movements')
        plt.legend(loc='best')
        plt.show()



class Binomial_Tree:
    def __init__(self,S,K,t,sigma,steps,r=0.05):
        self.K = K
        self.S = S
        self.t = t/365
        self.sigma = sigma
        self.r = r
        self.num_steps = steps

    def BT_Call_Option(self):
        #Time delt, up and down factor
        dT = self.t/self.num_steps
        u = np.exp(self.sigma*np.sqrt(dT))
        d = 1.0/u

        #Price vector
        V = np.zeros(self.num_steps+1)

        S_T = np.array([(self.S*u**j * d**(self.num_steps-j)) for j in range(self.num_steps+1)])

        a = np.exp(self.r*dT)
        p = (a-d)/(u-d)
        q = 1.0 - p

        V= np.maximum(S_T-self.K,0)

        for i in range(self.num_steps-1,-1,-1):
            V[:-1] = np.exp(-self.r*dT)*(p*V[1:]+q*V[:-1])

        return V[0]
    def BT_Put_Option(self):
        dT = self.t / self.num_steps
        u = np.exp(self.sigma * np.sqrt(dT))
        d = 1.0 / u

        # Price vector
        V = np.zeros(self.num_steps + 1)

        S_T = np.array([(self.S * u ** j * d ** (self.num_steps - j)) for j in range(self.num_steps + 1)])

        a = np.exp(self.r * dT)
        p = (a - d) / (u - d)
        q = 1.0 - p

        V = np.maximum(self.K - S_T, 0)

        for i in range(self.num_steps - 1, -1, -1):
            V[:-1] = np.exp(-self.r * dT) * (p * V[1:] + q * V[:-1])

        return V[0]




choice = st.sidebar.selectbox("Method",['Black-Scholes','Monte Carlo','Binomial Tree'])

S = st.sidebar.number_input("Spot Price ($)",value=100.00,min_value=0.00)
K = st.sidebar.number_input("Strike Price ($)",value=100.00,min_value=0.00)
sigma = st.sidebar.number_input("Volatility (σ)",value=0.20,min_value=0.00)
t = st.sidebar.number_input("Days to expiration",value=100,min_value=0)

r = st.sidebar.slider("Riskfree rate",min_value=0.0,max_value=0.10,step=0.01, value=0.05)
df =pd.DataFrame([[choice,S,K,t,r,sigma]], columns=['Type','Spot price','Strike price ','days to expiration','Risk-free rate','Volatility'])
st.table(df)



if choice == 'Black-Scholes':

    bsm = BlackScholes(S,K,t,sigma,r)
    call_option_bs = bsm.Black_Scholes_Call_Option()
    put_option_bs = bsm.Black_Scholes_Put_Option()

if choice == 'Monte Carlo':

    simulations = st.sidebar.number_input("Simulations",value=100,min_value=0)
    mc = Monte_Carlo(S,K,t,sigma,simulations,r)
    mc.simulate_prices()
    mc_call_option = mc.simulate_call()
    mc_put_option = mc.simulate_put()
    st.markdown("---")
    st.subheader("Random Walk")
    st.pyplot(mc.plot_simulation_results(100),clear_figure=False)

if choice == 'Binomial Tree':

    steps = st.sidebar.number_input("Number of steps",value=100,min_value=0)
    bt = Binomial_Tree(S,K,t,sigma,steps,r)
    bt_call_option = bt.BT_Call_Option()
    bt_put_option = bt.BT_Put_Option()



