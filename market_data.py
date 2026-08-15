import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import griddata


def get_spot_price(ticker):
    tk = yf.Ticker(ticker)
    try:
        price = tk.fast_info["last_price"]
        if price:
            return float(price)
    except Exception:
        pass
    hist = tk.history(period="5d")
    if hist.empty:
        raise ValueError(f"No price data found for '{ticker}'.")
    return float(hist["Close"].iloc[-1])


def get_expirations(ticker):
    return list(yf.Ticker(ticker).options)


def get_option_chain(ticker, expiration, option_type="call"):
    chain = yf.Ticker(ticker).option_chain(expiration)
    return chain.calls if option_type == "call" else chain.puts


def get_vol_surface_points(ticker, option_type="call", max_expirations=8,
                           moneyness_range=0.30, min_volume=0):
    tk = yf.Ticker(ticker)
    spot = get_spot_price(ticker)
    expirations = get_expirations(ticker)[:max_expirations]
    today = dt.date.today()

    rows = []
    for exp in expirations:
        try:
            df = get_option_chain(ticker, exp, option_type)
        except Exception:
            continue
        dte = (dt.datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if dte <= 0:
            continue
        for _, row in df.iterrows():
            iv = row.get("impliedVolatility", np.nan)
            strike = row.get("strike", np.nan)
            volume = row.get("volume", 0) or 0
            if not iv or np.isnan(iv) or iv <= 0:
                continue
            if abs(strike / spot - 1) > moneyness_range:
                continue
            if volume < min_volume:
                continue
            rows.append({
                "strike": strike,
                "dte": dte,
                "iv": iv * 100,
                "moneyness": strike / spot,
            })

    return spot, pd.DataFrame(rows)


def interpolate_surface(points, grid_size=60):
    strikes = points["strike"].values
    dtes = points["dte"].values
    ivs = points["iv"].values

    xi = np.linspace(strikes.min(), strikes.max(), grid_size)
    yi = np.linspace(dtes.min(), dtes.max(), grid_size)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((strikes, dtes), ivs, (Xi, Yi), method="cubic")
    nan_mask = np.isnan(Zi)
    if nan_mask.any():
        Zi_nearest = griddata((strikes, dtes), ivs, (Xi, Yi), method="nearest")
        Zi[nan_mask] = Zi_nearest[nan_mask]

    return Xi, Yi, Zi
