import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import market_data as md
from OptionPricing import BlackScholes, BinomialTree, MonteCarlo

st.set_page_config(
    page_title="Option Pricing Models",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Option Pricing Simulator")
st.markdown("---")

st.sidebar.header("Contract parameters")
S = st.sidebar.number_input("Spot price ($)", value=100.00, min_value=0.01)
K = st.sidebar.number_input("Strike price ($)", value=100.00, min_value=0.01)
sigma = st.sidebar.number_input("Volatility", value=0.20, min_value=0.01, step=0.01)
t = st.sidebar.number_input("Days to expiration", value=100, min_value=1)
r = st.sidebar.slider("Risk-free rate", min_value=0.0, max_value=0.10, step=0.005, value=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Model settings")
simulations = st.sidebar.number_input("Monte Carlo simulations", value=10_000, min_value=100, step=1000)
steps = st.sidebar.number_input("Binomial Tree steps", value=500, min_value=1, step=50)

pricing_tab, surface_tab = st.tabs(["Pricing & Greeks", "Volatility Surface"])

with pricing_tab:
    bs = BlackScholes(S, K, t, sigma, r)
    mc = MonteCarlo(S, K, t, sigma, simulations, r)
    bt = BinomialTree(S, K, t, sigma, steps, r)

    mc.simulate_prices()

    st.subheader("Estimated option prices")
    col_bs, col_mc, col_bt = st.columns(3)
    with col_bs:
        st.markdown("**Black-Scholes**")
        st.metric("Call", f"${bs.call_price():.4f}")
        st.metric("Put", f"${bs.put_price():.4f}")
    with col_mc:
        st.markdown(f"**Monte Carlo** ({int(simulations):,} paths)")
        st.metric("Call", f"${mc.call_price():.4f}")
        st.metric("Put", f"${mc.put_price():.4f}")
    with col_bt:
        st.markdown(f"**Binomial Tree** ({int(steps)} steps)")
        st.metric("Call", f"${bt.call_price():.4f}")
        st.metric("Put", f"${bt.put_price():.4f}")

    st.markdown("---")

    st.subheader("Option Greeks")
    greeks = bs.greeks()
    greeks_df = pd.DataFrame(greeks).T
    greeks_df = greeks_df[["delta", "gamma", "vega", "theta", "rho"]]
    greeks_df.columns = ["Delta", "Gamma", "Vega (per 1%)", "Theta (per day)", "Rho (per 1%)"]
    greeks_df.index = ["Call", "Put"]
    st.dataframe(greeks_df.style.format("{:.4f}"), use_container_width=True)

    with st.expander("Monte Carlo simulated price paths"):
        n_paths = st.slider("Paths to display", 1, min(int(simulations), 500), 50)
        st.pyplot(mc.plot_paths(n_paths))

with surface_tab:
    st.subheader("Implied volatility surface")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    ticker = c1.text_input("Ticker", value="AAPL").strip().upper()
    option_type = c2.selectbox("Option type", ["call", "put"])
    max_exp = c3.slider("Expirations", 2, 15, 8)
    moneyness = c4.slider("Moneyness range", 0.05, 0.60, 0.30, step=0.05)

    fetch = st.button("Build surface", type="primary")

    @st.cache_data(ttl=600, show_spinner="Fetching option chains")
    def load_surface(ticker, option_type, max_exp, moneyness):
        return md.get_vol_surface_points(
            ticker, option_type=option_type,
            max_expirations=max_exp, moneyness_range=moneyness,
        )

    if fetch and ticker:
        try:
            spot, points = load_surface(ticker, option_type, max_exp, moneyness)
        except Exception as exc:
            st.error(f"Could not load data for '{ticker}': {exc}")
            points = pd.DataFrame()

        if points.empty:
            st.warning("No usable option quotes returned. Try a more liquid ticker "
                       "or widen the moneyness range.")
        else:
            st.success(f"{ticker} spot: ${spot:,.2f} | {len(points)} quotes "
                       f"across {points['dte'].nunique()} expirations.")

            Xi, Yi, Zi = md.interpolate_surface(points)
            fig = go.Figure(data=[go.Surface(
                x=Xi, y=Yi, z=Zi, colorscale="Viridis",
                colorbar=dict(title="IV %"),
            )])
            fig.update_layout(
                height=650,
                scene=dict(
                    xaxis_title="Strike ($)",
                    yaxis_title="Days to expiry",
                    zaxis_title="Implied vol (%)",
                ),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw quotes"):
                st.dataframe(
                    points.sort_values(["dte", "strike"]).reset_index(drop=True),
                    use_container_width=True,
                )
    else:
        st.info("Enter a ticker and click Build surface.")
