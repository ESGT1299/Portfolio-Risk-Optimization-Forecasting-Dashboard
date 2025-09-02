# 📊 Portfolio Risk, Optimization & Forecasting Dashboard  

This project is a complete, end-to-end **quantitative finance analysis** using Python, GARCH models, portfolio optimization techniques, and Tableau for interactive visualization.  

It is designed to help analysts, students, and decision-makers **understand risk, optimize portfolios, and forecast future scenarios** using statistical and machine learning models — explained in the simplest way possible (Feynman style).  

---

## 🧠 Project Overview  

Imagine you want to invest money across several assets — stocks, bonds, crypto, gold, etc.  
You have three big questions:  

1. **Risk** – How risky are these assets?  
2. **Optimization** – How should I split my money across them to maximize returns for a given risk?  
3. **Forecasting** – What might happen in the future?  

This project answers these questions with a **step-by-step workflow**:

- ✅ Calculate **Risk KPIs** (Volatility, Sharpe, Sortino, Drawdown, VaR, CVaR)  
- ✅ Analyze **Correlations** between assets  
- ✅ Compute and visualize the **Efficient Frontier**  
- ✅ Find **Optimal Portfolio Weights** (Min Variance & Max Sharpe)  
- ✅ Forecast **Future Prices and Volatility** (ARIMA, Prophet, LSTM, GARCH)  
- ✅ Run **Monte Carlo Simulations** to model possible future scenarios  

---

## 🗂 Project Structure  


├── config.py

├── data_download.py

├── preprocessing.py

├── forecasting.py

├── garch.py

├── risk.py

├── optimization.py

├── monte_carlo.py

├── streamlit_app.py

├── (recommended) forecasting_export.py

├── (recommended) garch_export.py

├── data/

└── README.md


---

## 📚 Theory 

### 1️⃣ Risk Metrics  
- **Volatility (σ):** Measures how much prices move up and down — like a "nervousness meter".  
- **Sharpe Ratio:** Measures return per unit of risk — higher = better.  
- **Sortino Ratio:** Like Sharpe, but only penalizes downside risk (good for investors).  
- **Max Drawdown:** Largest drop from peak to trough — how bad things can get.  
- **VaR / CVaR:** "Worst-case" loss you might face with 95% confidence (VaR) and the **average** loss beyond that level (CVaR).

### 2️⃣ Correlation Matrix  
Correlation shows **how assets move together**:
- **+1** = move exactly together  
- **0** = unrelated  
- **-1** = move in opposite directions  

Diversification works when assets are not perfectly correlated.

### 3️⃣ Efficient Frontier  
The Efficient Frontier is like a "menu" of possible portfolios.  

It shows **the best return you can get for each level of risk**.  

Portfolios below the frontier are inefficient (you could do better).

### 4️⃣ Optimal Portfolios  
- **Min Variance Portfolio:** Smallest possible risk.

- **Max Sharpe Portfolio:** Best risk/return tradeoff (highest Sharpe ratio).  

### 5️⃣ Forecasting  
We forecast future prices with:  
- **ARIMA:** Statistical model for time series  
- **Prophet:** Seasonality/trend detection (by Facebook/Meta)  
- **LSTM:** Neural network model good for sequences  

We forecast volatility with **GARCH**, which models volatility clustering (when big moves follow big moves).

### 6️⃣ Monte Carlo Simulation  
Imagine running **10,000 possible futures** for your portfolio based on historical behavior.  
You get a "fan chart" showing likely ranges (p05, p25, p50, p75, p95).

---

## 🛠 Methodology (Step by Step)

1. **Data Collection & Cleaning**  
   - Imported price data, calculated returns, merged by asset and date.

2. **Risk Metrics Calculation**  
   - Used Python (pandas, numpy) to compute mean returns, volatilities, Sharpe, Sortino, VaR, CVaR.

3. **Correlation Matrix**  
   - Calculated pairwise correlations between assets and stored as a long-form table.

4. **Portfolio Optimization**  
   - Used mean-variance optimization to find weights that minimize risk and maximize Sharpe.

5. **Forecast Models**  
   - ARIMA, Prophet, and LSTM were trained on historical price data.  
   - GARCH was fit to historical volatility to predict future vol.

6. **Monte Carlo Simulation**  
   - Simulated thousands of future price paths.  
   - Calculated percentiles for fan chart visualization.

7. **Visualization**  
   - Imported all results into Tableau.  
   - Built **6 key plots and 2 dashboards**:
     - Risk KPIs Table  
     - Correlation Heatmap  
     - Efficient Frontier Scatter  
     - Optimal Portfolio Allocations  
     - Forecast Overlay (per model)  
     - Monte Carlo Percentile Fan  

---

## 📊 Tableau Dashboards  

- **Dashboard 1:** Risk & Portfolio Construction  
- **Dashboard 2:** Forecast & Simulation Analysis  

### Axis Names:
- **Date vs Price ($)** → for forecast overlay  
- **Date vs Volatility (σ)** → for GARCH vs realized  
- **Portfolio Volatility (σ) vs Portfolio Return** → Efficient Frontier  
- **Portfolio vs Weight (%)** → Optimal Allocations  
- **Simulation Day vs Value** → Monte Carlo Fan  

🌐 **Explore the Interactive Tableau Dashboard:**  
🔗 [Multi-Asset Risk, Correlation & Forecast Insights](https://public.tableau.com/views/Multi-AssetRiskCorrelationForecastInsights/RiskPortfolioConstruction?:language=es-ES&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## 🚀 How to Run Locally  

1. Clone the repository:

```bash
git clone https://github.com/ESGT1299/Portfolio-Risk-Optimization-Forecasting-Dashboard.git
cd portfolio-risk-forecast
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run the streamlit app

```bash
streamlit run streamlit_app.py
```

4. Open your browser and navigate to the local Streamlit server (usually http://localhost:8501).