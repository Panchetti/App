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
custom_css = """
<style>
.option-prices {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

.option-price {
    color: white;
    background-color: black;
    border: 2px solid white;
    padding: 10px;
    text-align: center;
    text-decoration: underline;
    font-size: 1.2em;
    margin: 0 10px; /* Space between boxes */
    width: 1500px; /* Set the width of the boxes */
    height: 120px; /* Set the height of the boxes */
    display: flex;
    align-items: center; /* Center text vertically */
    justify-content: center; /* Center text horizontally */
}
</style>
"""

# Add the CSS to your Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

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
    def __init__(self,S_0,K,t,sigma,simulations,r=0.05):
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
            return np.exp(-self.r*self.t)/self.N*np.sum(np.maximum(self.simulation_results_S[-1]-self.K,0))

    def simulate_put(self):
        if self.simulation_results_S is None:
            return -1
        else:
            return np.exp(-self.r*self.t)/self.N*np.sum(np.maximum(self.K-self.simulation_results_S[-1],0))

    def plot_simulation_results(self, num_of_movements):
        """Plots specified number of simulated price movements."""
        plt.figure(figsize=(12, 6))

        for i in range(num_of_movements):
            plt.plot(self.simulation_results_S[:, i], linestyle='-', color=f'C{i % 10}', alpha=0.7)

        plt.axhline(self.K, linestyle='-', label='Strike Price', linewidth=2)
        plt.title("Simulated Price Movements", fontsize=16, fontweight='bold')
        plt.xlabel("Days in Future", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='best', fontsize=10)

        plt.tight_layout()  # Adjust layout to make room for the labels
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





with st.sidebar:
    st.title("Option Pricing Models")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/kaloyan-panov-0734022a4/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Kaloyan Panov`</a>', unsafe_allow_html=True)

    choice = st.sidebar.selectbox("Method", ['Black-Scholes', 'Monte Carlo', 'Binomial Tree'])
    S = st.sidebar.number_input("Spot Price ($)",value=100.00,min_value=0.00)
    K = st.sidebar.number_input("Strike Price ($)",value=100.00,min_value=0.00)
    sigma = st.sidebar.number_input("Volatility (σ)",value=0.20,min_value=0.00)
    t = st.sidebar.number_input("Days to expiration",value=100,min_value=0)
    r = st.sidebar.slider("Riskfree rate",min_value=0.0,max_value=0.10,step=0.01, value=0.05)
    Put_price = 0.00
    Call_price = 0.00
    df =pd.DataFrame([[choice,float(S),float(K),t,r,sigma]], columns=['Type','Spot price','Strike price ','days to expiration','Risk-free rate','Volatility'])
    if choice == "Black-Scholes":
        calculate = st.sidebar.button('Simulate Prices')

df['Risk-Free rate'] = df['Risk-free rate'].map("{:.2f}".format)
df['Volatility'] = df['Volatility'].map("{:.2f}".format)
df['Spot price'] = df['Spot price'].map("{:.2f}".format)
df['Strike price '] = df['Strike price '].map("{:.2f}".format)
html = df.to_html(index=False, escape=False)

st.markdown(
    f'<div style="overflow-x:auto; text-align: center; width :100%;">{html}</div>',
    unsafe_allow_html=True
)

st.markdown("---")
(st.subheader("Option Price"))



if choice == 'Black-Scholes' and  calculate == True:

    bsm = BlackScholes(S,K,t,sigma,r)
    Call_price = bsm.Black_Scholes_Call_Option()
    Put_price = bsm.Black_Scholes_Put_Option()

if choice == 'Monte Carlo':

    simulations = st.sidebar.number_input("Simulations", value=100, min_value=0)
    calculate = st.sidebar.button('Simulate Prices')
    mc = Monte_Carlo(S, K, t, sigma, simulations, r)


    if calculate == True:

        mc.simulate_prices()
        Call_price = mc.simulate_call()
        Put_price = mc.simulate_put()




if choice == 'Binomial Tree':

    steps = st.sidebar.number_input("Number of steps",value=100,min_value=0)
    calculate = st.sidebar.button('Simulate Prices')

    if calculate == True:

        bt = Binomial_Tree(S,K,t,sigma,steps,r)
        Call_price = bt.BT_Call_Option()
        Put_price = bt.BT_Put_Option()



st.markdown(
            f"<div class='option-prices'>"
            f"<div class='option-price'>Call Option Price: ${float(Call_price):.3f}</div>"
            f"<div class='option-price'>Put Option Price: ${float(Put_price):.3f}</div>"
            "</div>", unsafe_allow_html=True
        )

st.markdown('---')
st.subheader("P&L Scenario")
pnl_df = pd.DataFrame(0,columns=['-30%', '-20%', '-10%', '0%', '10%', '20%', '30%'],
                       index=['Underlying', 'Call PnL', 'Put PnL'])
# Populate the DataFrame
if calculate == True:
    for i, scen in enumerate(range(-30, 31, 10)):
        # Calculate the new price based on the percentage change
        new_price = S * (1 + scen / 100)

        # Calculate Call PnL
        call_pnl = (max(new_price - K, 0) - float(Call_price))*100
        # Calculate Put PnL
        put_pnl = (max(K - new_price, 0) - float(Put_price))*100

        # Set values in the DataFrame
        pnl_df.iloc[0, i] = new_price  # Underlying price
        pnl_df.iloc[1, i] = call_pnl  # Call PnL
        pnl_df.iloc[2, i] = put_pnl


st.table(pnl_df)



# Prepare data for the line charts
call_pnl_data = pnl_df.loc['Call PnL'].astype(float)
put_pnl_data = pnl_df.loc['Put PnL'].astype(float)

# Reorder the data
ordered_indices = ['-30%', '-20%', '-10%', '0%', '10%', '20%', '30%']
call_pnl_data = call_pnl_data[ordered_indices].reset_index(drop=True)
put_pnl_data = put_pnl_data[ordered_indices].reset_index(drop=True)

# Create side-by-side columns for the line charts
col1, col2 = st.columns(2)

with col1:
    st.subheader('Call Option Profit and Loss')
    st.line_chart(call_pnl_data)

with col2:
    st.subheader('Put Option Profit and Loss')
    st.line_chart(put_pnl_data)
if choice == 'Monte Carlo' and calculate == True:
    st.markdown("---")
    st.subheader("Random Walk")
    st.pyplot(mc.plot_simulation_results(100), clear_figure=False)
