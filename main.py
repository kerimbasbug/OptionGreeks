import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import base64

from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega

from OptionPlotter import OptionPlotter

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

class OptionAnalysis:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.stock_last_close = self.stock.history(period="1d")["Close"].iloc[-1]
        self.rf_rate = None
        self.expiration_date = None
        self.days_difference = None
        self.calls = None
        self.puts = None
        self.strategy = None
        self.strike_selection = None
        self.sigma = None
        self.premium_option = None
        self.greeks_df = None
        self.greek_selection = None

    def get_rf_rate(self):
        #st.write(yf.__version__)
        self.rf_rate = yf.download('^TNX', end=datetime.today(), progress=False)['Close'].iloc[-1].values[0]/100

    def select_expiration_date(self):
        strike_dates = self.stock.options
        strike_dates = [datetime.strptime(date, "%Y-%m-%d") for date in strike_dates]
        strike_dates = [str(date.date()) for date in strike_dates if (date - datetime.today()).days > 0] # Filter valid strike dates
        self.expiration_date = st.sidebar.selectbox("Expiration Date", strike_dates, index=0)
        option_chain = self.stock.option_chain(self.expiration_date)
        self.calls = option_chain.calls
        self.puts = option_chain.puts

    def select_strike_price(self):
        closest_strike = min(self.calls['strike'], key=lambda x: abs(x - self.stock_last_close))
        if 'Call' in self.strategy:
            ind = int(self.calls[self.calls['strike'] == closest_strike].index[0])
            self.strike_selection = st.sidebar.selectbox(
                "Strike Price",
                self.calls['strike'].iloc[np.maximum(0,ind-5):ind+6],
                index=5
            )
            self.premium_option = self.calls[self.calls['strike'] == self.strike_selection]['lastPrice'].iloc[0]
        else:
            ind = int(self.puts[self.puts['strike'] == closest_strike].index[0])
            self.strike_selection = st.sidebar.selectbox(
                "Strike Price",
                self.puts['strike'].iloc[np.maximum(0,ind-5):ind+6],
                index=5
            )
            self.premium_option = self.puts[self.puts['strike'] == self.strike_selection]['lastPrice'].iloc[0]

    def calculate_days_difference(self):
        self.days_difference = (datetime.strptime(self.expiration_date, "%Y-%m-%d") - datetime.today()).days

    def update_markdown(self):
        company_df = pd.read_csv('Data/companies_info.csv')
        company_name = company_df[company_df['ticker']==self.ticker]['short name'].values[0]
        company_sector = company_df[company_df['ticker']==self.ticker]['sector'].values[0]
        company_logo = encode_image_to_base64("Logos/"+self.ticker+".png")

        stock_data = self.stock.history(period="3mo").reset_index(drop=False)
        last_day_return = (stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]
        plot_color = 'green' if last_day_return >=0 else 'red'

        plt.figure(figsize=(2.5, 1.5))
        plt.plot(stock_data['Date'], stock_data['Close'], color=plot_color, label='Stock Price')
        plt.fill_between(stock_data['Date'], stock_data['Close'], color=plot_color, alpha=0.25)
        plt.ylim(stock_data['Close'].min()*0.9, stock_data['Close'].max()*1.1)
        plt.axis('off')
        plt.savefig('stock_plot.png', bbox_inches='tight', pad_inches=0)

        plot_base64 = encode_image_to_base64('stock_plot.png')
        st.markdown(
            f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; color: rgba(54, 53, 53, 0.8); font-family: Arial; text-align: left; max-width: 800px; margin: 0px auto; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);">
                <div style="display: flex; align-items: center;">
                    <img src="{company_logo}" style="max-width: 80px; max-height: 80px; margin-right: 30px; object-fit: contain;">
                    <div>
                        <h3 style="margin: 0; padding: 0; line-height: 1.4;">{company_name}</h3>
                        <h4 style="margin: 0; padding: 0; line-height: 1.4;">{self.ticker}</h4>
                        <h7 style="margin: 0; padding: 0; line-height: 1.2;">{company_sector}</h7>
                    </div>
                    <img src="{plot_base64}" style="max-width: 300px; max-height: 200px; position: absolute; top: 10px; right: 30px; object-fit: contain;">
                </div>
                <hr style="border: none; border-top: 1px solid #ccc; margin: 10px 0;">
                <p style="margin: 0; text-align: center; display: flex; justify-content: space-evenly; align-items: center;">
                    <span style="display: flex; align-items: center;">
                        <b>Last Close Price:</b><span style="margin-left: 2px;">${self.stock_last_close:.1f}&nbsp;&nbsp;</span>
                        <span style="color: {plot_color};">
                              ({100 * last_day_return:.2f}%)
                        </span>
                    </span>
                    <span style="display: flex; align-items: center;">
                        <b>Risk Free Rate:</b><span style="margin-left: 2px;">{self.rf_rate*100:.2f}%</span>
                    </span>
                 </p>
                <hr style="border: none; border-top: 1px solid #ccc; margin: 10px 0;">
                <p style="margin: 0; text-align: center; display: flex; justify-content: space-evenly; align-items: center;">
                    <span style="display: flex; align-items: center;">
                        <b>Strike:</b><span style="margin-left: 2px;">${self.strike_selection}</span>
                    </span>
                    <span style="display: flex; align-items: center;">
                        <b>Premium:</b><span style="margin-left: 2px;">${self.premium_option}</span>
                    </span>
                    <span style="display: flex; align-items: center;">
                        <b>Days Till Expiration:</b><span style="margin-left: 2px;">{self.days_difference}</span>
                    </span>
                    <span style="display: flex; align-items: center;">
                        <b>IV:</b><span style="margin-left: 2px;">%{self.sigma*100:.2f}</span>
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    def select_strategy(self):
        self.strategy = st.sidebar.radio(
            "Strategy",
            [
                "Long Call :green[Bullish]", 
                "Short Call :red[Bearish]",
                "Long Put :red[Bearish]", 
                "Short Put :green[Bullish]"
            ],
        )
    def select_greek(self):
        greek_map = {
            "Delta": "Delta",
            "Gamma": "Gamma",
            "Rho": "Rho",
            "Theta": "Theta",
            "Vega": "Vega"
        }
        st.markdown(
            """
            <style>
                .stButtonGroup {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        with st.container():
            st.markdown('<div class="stButtonGroup">', unsafe_allow_html=True)
            self.greek_selection = st.segmented_control(
                "",
                options=greek_map.keys(),
                format_func=lambda option: greek_map[option],
                selection_mode="single",
                default="Delta"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    def display_option_table(self):
        if 'Call' in self.strategy:
            st.table(self.calls[self.calls['strike'] == self.strike_selection])
        else:
            st.table(self.puts[self.puts['strike'] == self.strike_selection])

    def implied_vol(self, S0, K, T, r, market_price, flag='c', tol=0.00001):
        max_iter = 200
        vol_old = 0.30

        for i in range(max_iter):
            bs_price = bs(flag, S0, K, T, r, vol_old)
            Cprime =  vega(flag, S0, K, T, r, vol_old)*100
            C = bs_price - market_price
            vol_new = vol_old - C/Cprime
            bs_new = bs(flag, S0, K, T, r, vol_new)

            if (abs(vol_old - vol_new) < tol or abs(bs_new - market_price) < tol):
                break
            vol_old = vol_new

        implied_vol = vol_old
        return implied_vol

    def fetch_and_set_iv(self, threshold = 0.00001):
        if 'Call' in self.strategy:
            closest_strike = min(self.calls['strike'], key=lambda x: abs(x - self.stock_last_close))
            ind = int(self.calls[self.calls['strike'] == closest_strike].index[0])
            df_iv = self.calls[:].iloc[np.maximum(0,ind-10):ind+9][['strike', 'lastPrice']]
            df_iv['iv'] = df_iv.apply(lambda x: self.implied_vol(S0=self.stock_last_close, K=x['strike'], T=np.maximum(0,self.days_difference/252), r=self.rf_rate, market_price=x['lastPrice'], flag='c'), axis=1)
            df_iv['impvol'] = df_iv['iv'].apply(lambda x: float('nan') if x <= threshold else x)  # Replace values below threshold with NaN
            df_iv['impvol'] = df_iv['impvol'].interpolate(method='linear')  # Interpolate missing values
            df_iv['impvol'] = df_iv['impvol'].fillna(method='bfill')
            #st.table(df_iv)
            iv = df_iv[df_iv['strike']==self.strike_selection]['impvol'].values[0]
            if not np.isnan(iv) and iv>0:
                self.sigma = iv

        elif 'Put' in self.strategy:
            closest_strike = min(self.puts['strike'], key=lambda x: abs(x - self.stock_last_close))
            ind = int(self.puts[self.puts['strike'] == closest_strike].index[0])
            df_iv = self.puts[:].iloc[np.maximum(0,ind-10):ind+9][['strike', 'lastPrice']]
            df_iv['iv'] = df_iv.apply(lambda x: self.implied_vol(S0=self.stock_last_close, K=x['strike'], T=np.maximum(0,self.days_difference/252), r=self.rf_rate, market_price=x['lastPrice'], flag='p'), axis=1)
            df_iv['impvol'] = df_iv['iv'].apply(lambda x: float('nan') if x <= threshold else x)  # Replace values below threshold with NaN
            df_iv['impvol'] = df_iv['impvol'].interpolate(method='linear')  # Interpolate missing values
            df_iv['impvol'] = df_iv['impvol'].fillna(method='bfill')
            #st.table(df_iv)
            iv = df_iv[df_iv['strike']==self.strike_selection]['impvol'].values[0]
            if not np.isnan(iv) and iv>0:
                self.sigma = iv

    def calculate_greeks(self, S_list, K, T, r, sigma, q=0, option_type='Call', position='Long'):
        results = []
        position_multiplier = 1 if position == 'Long' else -1 # Multiplier for position: 1 for Long, -1 for Short
        
        for S in S_list:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == 'Call':
                delta = np.exp(-q * T) * norm.cdf(d1)
            else:
                delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
            delta *= position_multiplier

            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            gamma *= position_multiplier

            if option_type == 'Call':
                theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                        - r * K * np.exp(-r * T) * norm.cdf(d2)
                        + q * S * np.exp(-q * T) * norm.cdf(d1))
            else:
                theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)
                        - q * S * np.exp(-q * T) * norm.cdf(-d1))
            theta = (theta / 365) * position_multiplier

            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
            vega *= position_multiplier

            if option_type == 'Call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            rho *= position_multiplier

            results.append({
                'Underlying Price': S,
                'Delta': np.round(delta, 5),
                'Gamma': np.round(gamma, 5),
                'Theta': np.round(theta, 5),
                'Vega': np.round(vega, 5),
                'Rho': np.round(rho, 5)
            })

        return pd.DataFrame(results)
    
    def get_greeks(self):
        S_list = list(np.arange(0.0001, self.stock_last_close * 2, 0.1))
        if 'Call' in self.strategy:
            if 'Long' in self.strategy:
                self.greeks_df = self.calculate_greeks(
                    S_list,
                    K=self.strike_selection,
                    T=self.days_difference / 365,
                    r=self.rf_rate,
                    sigma=self.sigma,
                    q=0,
                    option_type='Call',
                    position='Long'
                )
            else:
                self.greeks_df = self.calculate_greeks(
                    S_list,
                    K=self.strike_selection,
                    T=self.days_difference / 365,
                    r=self.rf_rate,
                    sigma=self.sigma,
                    q=0,
                    option_type='Call',
                    position='Short'
                )
        elif 'Put' in self.strategy:
            if 'Long' in self.strategy:
                self.greeks_df = self.calculate_greeks(
                    S_list,
                    K=self.strike_selection,
                    T=self.days_difference / 365,
                    r=self.rf_rate,
                    sigma=self.sigma,
                    q=0,
                    option_type='Put',
                    position='Long'
                )
            else:
                self.greeks_df = self.calculate_greeks(
                    S_list,
                    K=self.strike_selection,
                    T=self.days_difference / 365,
                    r=self.rf_rate,
                    sigma=self.sigma,
                    q=0,
                    option_type='Put',
                    position='Short'
                )

    def calculate_pnl(self):
        if 'Long Call' in self.strategy:
            self.greeks_df['P&L'] = np.maximum(self.greeks_df['Underlying Price'] - self.strike_selection, 0) - self.premium_option
        elif 'Short Call' in self.strategy:
            self.greeks_df['P&L'] = self.premium_option - np.maximum(self.greeks_df['Underlying Price'] - self.strike_selection, 0)
        elif 'Long Put' in self.strategy:
            self.greeks_df['P&L'] = np.maximum(self.strike_selection - self.greeks_df['Underlying Price'], 0) - self.premium_option
        elif 'Short Put' in self.strategy:
            self.greeks_df['P&L'] = self.premium_option - np.maximum(self.strike_selection - self.greeks_df['Underlying Price'], 0)

    def plot_data(self):
        plotter = OptionPlotter(self.greeks_df, self.strike_selection, self.premium_option, self.greek_selection, self.stock_last_close)

        if 'Long Call' in self.strategy:
            fig = plotter.plot_long_call()
        elif 'Short Call' in self.strategy:
            fig = plotter.plot_short_call()
        elif 'Long Put' in self.strategy:
            fig = plotter.plot_long_put()
        elif 'Short Put' in self.strategy:
            fig = plotter.plot_short_put()
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    def run_analysis(self):
        self.select_strategy()
        self.get_rf_rate()
        self.select_expiration_date()
        self.select_strike_price()
        self.calculate_days_difference()
        self.fetch_and_set_iv()
        self.update_markdown()
        self.select_greek()
        self.get_greeks()
        self.calculate_pnl()
        self.plot_data()


st.title("Options P&L and Greeks")
st.sidebar.title('Option Parameters')
sp100_tickers = pd.read_csv('Data/companies_info.csv')['ticker']
ticker = st.sidebar.selectbox("Ticker", sorted(sp100_tickers), index=0)
option_analysis = OptionAnalysis(ticker)
option_analysis.run_analysis()

st.info("""
##### Disclaimer
This tool provides profit and loss calculations for options based on data sourced from Yahoo Finance. 
Please note that the accuracy and reliability of the results depend on the completeness and correctness 
of the data retrieved. This application is intended for educational and informational purposes only and 
should not be relied upon for trading or investment decisions. Always verify the data and consult a 
professional financial advisor before making any financial decisions.
""")

st.sidebar.markdown("---")
st.sidebar.write("Created by: Kerim Başbuğ")

github_logo = encode_image_to_base64('Icons/github.png')
linkedin_logo = encode_image_to_base64('Icons/linkedin.png')

st.sidebar.markdown(f'<a href="https://github.com/kerimbasbug" target="_blank" style="text-decoration: none; color: inherit;"><img src="{github_logo}" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`kerimbasbug`</a>', unsafe_allow_html=True)
st.sidebar.markdown(f'<a href="https://www.linkedin.com/in/kerimbasbug/" target="_blank" style="text-decoration: none; color: inherit;"><img src="{linkedin_logo}" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`kerimbasbug`</a>', unsafe_allow_html=True)

