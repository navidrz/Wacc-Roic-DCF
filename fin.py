import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Any, Dict, Optional
import logging
from datetime import datetime

# -----------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="💡 Financial Calculators",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': ("💡 **Financial Calculators App**\n\n"
                  "Use this app to calculate key financial metrics:\n"
                  "- **ROIC (Return on Invested Capital)**\n"
                  "- **WACC (Weighted Average Cost of Capital)**\n"
                  "- **DCF (Discounted Cash Flow)**\n\n"
                  "All calculations follow standard corporate finance theory:\n"
                  "- *Brealey, Myers & Allen - Principles of Corporate Finance*\n"
                  "- *McKinsey & Company - Valuation: Measuring and Managing the Value of Companies*\n"
                  "This tool is for educational and analytical purposes only.")
    }
)

# -----------------------------------------------------------
# Initialize Logging
# -----------------------------------------------------------
logging.basicConfig(
    filename='financial_calculators.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def format_number(num: float, currency_symbol: str = "") -> str:
    """Format a number with commas and two decimal places."""
    try:
        return f"{currency_symbol}{num:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(num: float) -> str:
    """Format a number as a percentage with two decimal places."""
    try:
        return f"{num:.2f}%"
    except (ValueError, TypeError):
        return "N/A"

def export_results_as_csv(dataframe: pd.DataFrame, filename: str, label: str):
    """
    Provide a download button for DataFrames as CSV files.
    """
    try:
        csv = dataframe.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"❌ Error exporting CSV: {e}")
        logging.error(f"CSV Export Error: {e}")

# -----------------------------------------------------------
# Financial Data Handler
# -----------------------------------------------------------
class FinancialDataHandler:
    """
    Handles loading, validating, and processing of financial data from uploaded Excel files.
    Provides functionalities to parse, retrieve, and use data for calculations.
    """
    def __init__(self, financial_data: Dict[str, pd.DataFrame]):
        self.financial_data = financial_data
        self.periods = self.get_all_periods()

    def get_all_periods(self) -> list:
        """Retrieve a sorted list of all financial periods available."""
        try:
            bs_periods = list(self.financial_data['balance_sheet'].index)
            sorted_periods = sorted(bs_periods, key=lambda x: self.parse_period(x), reverse=True)
            return sorted_periods
        except KeyError as e:
            logging.error(f"Missing financial statement for periods: {e}")
            return []

    def parse_period(self, period_str: str) -> datetime:
        """Parse a period string to datetime for sorting (expected format: 'FY-ending YYYY-MM')."""
        try:
            date_str = period_str.split('ending')[-1].strip()
            return datetime.strptime(date_str, '%Y-%m')
        except Exception as e:
            logging.error(f"Error parsing period '{period_str}': {e}")
            return datetime.min

    def get_latest_financial_period(self) -> Optional[str]:
        """Retrieve the latest financial period."""
        return self.periods[0] if self.periods else None

    def get_financials_for_period(self, period: str) -> Dict[str, pd.Series]:
        """Retrieve financial statements for a specific period."""
        try:
            bs = self.financial_data['balance_sheet'].loc[period]
            cf = self.financial_data['cashflow'].loc[period]
            is_ = self.financial_data['income_statement'].loc[period]
            ks = self.financial_data['keystats'].loc[period]
            return {
                'balance_sheet': bs,
                'cashflow': cf,
                'income_statement': is_,
                'keystats': ks
            }
        except KeyError as e:
            logging.error(f"Financial period '{period}' not found: {e}")
            return {
                'balance_sheet': pd.Series(),
                'cashflow': pd.Series(),
                'income_statement': pd.Series(),
                'keystats': pd.Series()
            }

    def calculate_beta(self) -> Optional[float]:
        """
        Calculate Beta using historical stock prices and market index data.
        Reference: CAPM model from standard finance theory.
        """
        try:
            df_stock = self.financial_data.get('historical_prices', pd.DataFrame())
            df_market = self.financial_data.get('market_index', pd.DataFrame())
            df_risk_free = self.financial_data.get('risk_free_rate', pd.DataFrame())

            if df_stock.empty or df_market.empty or df_risk_free.empty:
                st.warning("⚠️ Insufficient data to calculate Beta. Please ensure all historical data is provided.")
                return None

            # Rename columns
            df_market = df_market.rename(columns={'تاریخ میلادی': 'Date_Gregorian', 'مقدار': 'Market_Index'})
            df_risk_free = df_risk_free.rename(columns={'date': 'Date_Gregorian', 'ytm': 'Risk_Free_Rate'})

            # Merge datasets on Date
            df = pd.merge(df_stock[['Date', 'Close']], df_market[['Date_Gregorian', 'Market_Index']],
                          left_on='Date', right_on='Date_Gregorian', how='inner')
            df = pd.merge(df, df_risk_free[['Date_Gregorian', 'Risk_Free_Rate']], on='Date_Gregorian', how='inner')

            if df.empty:
                st.warning("⚠️ No overlapping dates between stock prices, market index, and risk-free rate.")
                return None

            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Convert columns to numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce').fillna(0)
            df['Market_Index'] = pd.to_numeric(df['Market_Index'], errors='coerce').fillna(0)
            df['Risk_Free_Rate'] = pd.to_numeric(df['Risk_Free_Rate'], errors='coerce').fillna(0)

            # Calculate daily returns
            df['Stock_Return'] = df['Close'].pct_change()
            df['Market_Return'] = df['Market_Index'].pct_change()

            # Assuming daily risk-free is already in daily terms
            df['Risk_Free_Return'] = df['Risk_Free_Rate'] / 100
            df.dropna(inplace=True)

            if df.empty:
                st.warning("⚠️ Insufficient data after calculating daily returns.")
                return None

            # Excess Returns
            df['Excess_Stock_Return'] = df['Stock_Return'] - df['Risk_Free_Return']
            df['Excess_Market_Return'] = df['Market_Return'] - df['Risk_Free_Return']

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=['Excess_Stock_Return', 'Excess_Market_Return'], inplace=True)

            if df['Excess_Market_Return'].nunique() <= 1 or df['Excess_Stock_Return'].nunique() <= 1:
                st.error("❌ Insufficient variation in returns to perform regression for Beta.")
                return None

            # Display sample of processed data
            st.markdown("### 📊 Processed Data for Beta Calculation")
            st.dataframe(df[['Date', 'Excess_Market_Return', 'Excess_Stock_Return']].head())

            # Linear regression for Beta
            X = df['Excess_Market_Return'].values.reshape(-1, 1)
            y = df['Excess_Stock_Return'].values
            reg = LinearRegression()
            reg.fit(X, y)
            beta = reg.coef_[0]
            r_squared = reg.score(X, y)

            # Plot regression
            fig = px.scatter(
                x=df['Excess_Market_Return'], y=df['Excess_Stock_Return'],
                labels={'x': 'Excess Market Return', 'y': 'Excess Stock Return'},
                title='📈 Excess Stock Returns vs. Excess Market Returns'
            )
            x_range = np.linspace(df['Excess_Market_Return'].min(), df['Excess_Market_Return'].max(), 100)
            y_pred = reg.predict(x_range.reshape(-1, 1))
            fig.add_traces(px.line(x=x_range, y=y_pred).data)

            # Annotation
            fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper",
                               text=f"Beta: {beta:.4f}<br>R²: {r_squared:.4f}",
                               showarrow=False, align='left', bgcolor="rgba(0,0,0,0.5)",
                               font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

            return beta
        except Exception as e:
            st.error(f"❌ Error calculating Beta: {e}")
            logging.error(f"Beta Calculation Error: {e}")
            return None

# -----------------------------------------------------------
# Financial Calculators
# -----------------------------------------------------------

def roic_calculator(financials: Dict[str, pd.DataFrame]):
    """
    Calculate Return on Invested Capital (ROIC).
    ROIC = NOPAT / Invested Capital
    Reference: McKinsey Valuation & standard corporate finance literature.
    """
    st.header("📈 ROIC Calculator")
    st.markdown("""
    **Return on Invested Capital (ROIC)** measures how effectively a company uses its capital to generate returns.
    \n**Formula:** ROIC = NOPAT / Invested Capital
    \n*Reference: McKinsey & Company - Valuation: Measuring and Managing the Value of Companies.*
    """)

    handler = FinancialDataHandler(financials)
    latest_period = handler.get_latest_financial_period()

    if not latest_period:
        st.error("❌ No financial period data available. Please upload and load your data.")
        return

    with st.form("roic_form"):
        st.subheader("📋 Input Parameters")
        tax_rate = st.number_input("Tax Rate (%) for ROIC:", min_value=0.0, value=21.0, step=0.1,
                                   help="Corporate tax rate used to calculate NOPAT.")
        period = st.selectbox("Select Financial Period for ROIC:", options=handler.periods,
                              help="Choose the period for which you want to calculate ROIC.")
        submit = st.form_submit_button("Calculate ROIC")

    if submit:
        try:
            financials_period = handler.get_financials_for_period(period)
            balance_sheet = financials_period['balance_sheet']
            income_statement = financials_period['income_statement']

            if balance_sheet.empty or income_statement.empty:
                st.error(f"❌ Incomplete data for the period {period}. Please ensure full financial statements are provided.")
                return

            # Check if company is a bank for debt calculation logic
            bank_indicators = [
                'Customers Deposits',
                'Credit Facilities',
                'Credit Facilities to Governmental Entities',
                'Due to Banks and Credit Institutions',
                'Term Deposits'
            ]
            is_bank = any(indicator in balance_sheet.index for indicator in bank_indicators)
            interest_paid = balance_sheet.get('Interest Paid for Borrowing', 0)
            is_islamic = is_bank and (interest_paid == 0)

            # Calculate Total Debt
            if is_bank:
                # Islamic finance conventions
                if is_islamic:
                    debt_fields = [
                        'Debt securities',
                        'Customers Deposits',
                        'Term Deposits',
                        'Due to Banks and Credit Institutions',
                        'Credit Facilities',
                        'Credit Facilities to Governmental Entities'
                    ]
                    existing_debt_fields = [field for field in debt_fields if field in balance_sheet.index]
                    total_debt = balance_sheet[existing_debt_fields].sum()
                else:
                    total_debt = balance_sheet.get('Total Liabilities', 0)
            else:
                current_portion = balance_sheet.get('Current Portion of Loan Payable', 0)
                long_term_debt = balance_sheet.get('Long Term Debt', 0)
                total_debt = current_portion + long_term_debt

            # Convert to float safely
            total_debt = float(total_debt) if pd.notnull(total_debt) else 0.0
            total_equity = float(balance_sheet.get("Total Stockholders' Equity", 0) or 0.0)
            cash = float(balance_sheet.get("Cash", 0) or 0.0)

            # Invested Capital
            invested_capital = total_debt + total_equity - cash
            if invested_capital <= 0:
                st.error("⚠️ Invested Capital is zero or negative. Cannot compute ROIC.")
                return

            # Operating Income (EBIT)
            operating_income = float(income_statement.get('Operating Profit (Loss)', 0) or 0.0)
            tax_rate_decimal = tax_rate / 100
            nopat = operating_income * (1 - tax_rate_decimal)
            roic = (nopat / invested_capital) * 100 if invested_capital != 0 else 0

            # Display Results
            st.subheader(f"📊 ROIC: {format_percentage(roic)}")
            with st.expander("🔍 Calculation Details"):
                details = {
                    "Total Debt": format_number(total_debt),
                    "Total Equity": format_number(total_equity),
                    "Cash": format_number(cash),
                    "Invested Capital": format_number(invested_capital),
                    "Operating Income (EBIT)": format_number(operating_income),
                    "NOPAT": format_number(nopat),
                    "ROIC (%)": format_percentage(roic)
                }
                st.table(pd.DataFrame(list(details.items()), columns=["Parameter", "Value"]))

            # Historical ROIC Trend (last 5 periods)
            roic_values = []
            for yr in handler.periods[:5]:
                fin_period = handler.get_financials_for_period(yr)
                bs = fin_period['balance_sheet']
                is_ = fin_period['income_statement']

                if bs.empty or is_.empty:
                    continue

                is_bank_yr = any(indicator in bs.index for indicator in bank_indicators)
                interest_paid_yr = bs.get('Interest Paid for Borrowing', 0)
                is_islamic_yr = is_bank_yr and (interest_paid_yr == 0)

                if is_bank_yr:
                    if is_islamic_yr:
                        debt_fields_yr = [
                            'Debt securities',
                            'Customers Deposits',
                            'Term Deposits',
                            'Due to Banks and Credit Institutions',
                            'Credit Facilities',
                            'Credit Facilities to Governmental Entities'
                        ]
                        existing_debt_fields_yr = [field for field in debt_fields_yr if field in bs.index]
                        total_debt_yr = bs[existing_debt_fields_yr].sum()
                    else:
                        total_debt_yr = bs.get('Total Liabilities', 0)
                else:
                    current_portion_yr = bs.get('Current Portion of Loan Payable', 0)
                    long_term_debt_yr = bs.get('Long Term Debt', 0)
                    total_debt_yr = current_portion_yr + long_term_debt_yr

                total_debt_yr = float(total_debt_yr or 0.0)
                total_equity_yr = float(bs.get("Total Stockholders' Equity", 0) or 0.0)
                cash_yr = float(bs.get("Cash", 0) or 0.0)

                invested_capital_yr = total_debt_yr + total_equity_yr - cash_yr
                if invested_capital_yr <= 0:
                    continue

                operating_income_yr = float(is_.get('Operating Profit (Loss)', 0) or 0.0)
                nopat_yr = operating_income_yr * (1 - tax_rate_decimal)
                roic_yr = (nopat_yr / invested_capital_yr) * 100 if invested_capital_yr != 0 else 0
                roic_values.append({"Period": yr, "ROIC (%)": roic_yr})

            if roic_values:
                roic_df = pd.DataFrame(roic_values)
                fig = px.line(roic_df, x='Period', y='ROIC (%)', title='📈 ROIC Over Periods', markers=True)
                st.plotly_chart(fig, use_container_width=True)
                export_results_as_csv(roic_df, 'ROIC_Over_Periods.csv', "Download ROIC Over Periods as CSV")
            else:
                st.info("⚠️ Not enough data to plot ROIC over periods.")

        except Exception as e:
            st.error(f"❌ Error in ROIC calculation: {e}")
            logging.error(f"ROIC Calculator Error: {e}")
    st.markdown("---")


def wacc_calculator(financials: Dict[str, pd.DataFrame]):
    """
    Calculate Weighted Average Cost of Capital (WACC).
    WACC = (E/(E+D))*Re + (D/(E+D))*Rd*(1-T)
    References: Brealey, Myers & Allen - Principles of Corporate Finance
                McKinsey Valuation
    """
    st.header("💼 WACC Calculator")
    st.markdown("""
    **WACC (Weighted Average Cost of Capital)** is the firm's overall required return on its sources of capital.
    \n**Formula:** WACC = (E/(E+D))*Re + (D/(E+D))*Rd*(1 - Tc)
    \n*Reference: Brealey, Myers & Allen - Principles of Corporate Finance.*
    """)

    currency_options = ["IRR", "USD", "EUR", "GBP"]
    currency = st.selectbox("Select Currency for Outputs:", options=currency_options, index=0, key='wacc_currency',
                            help="Choose the currency symbol for WACC outputs.")
    currency_symbols = {
        "IRR": "IRR ",
        "USD": "$",
        "EUR": "€",
        "GBP": "£"
    }
    currency_symbol = currency_symbols.get(currency, "IRR ")

    handler = FinancialDataHandler(financials)
    latest_period = handler.get_latest_financial_period()

    if not latest_period:
        st.error("❌ No financial period data available. Please upload and load your data.")
        return

    # Store the number of outstanding shares outside the form to avoid reset
    # Adding a key to preserve state
    outstanding_shares = st.number_input(
        "Number of Outstanding Shares:",
        min_value=0.0, value=3600000000.0, step=1000000.0,
        help="Enter the total number of outstanding shares to estimate market capitalization.",
        key="outstanding_shares"
    )

    with st.form("wacc_form"):
        st.subheader("📋 Input Parameters")
        risk_free_rate_input = st.number_input("Risk-Free Rate (Rf) (%) :", min_value=0.0, value=3.0, step=0.1,
                                               help="A typically long-term government bond yield.")
        market_return_input = st.number_input("Expected Market Return (Rm) (%) :", min_value=0.0, value=8.0, step=0.1,
                                              help="Overall expected return of the market portfolio.")
        tax_rate = st.number_input("Corporate Tax Rate (Tc) (%) :", min_value=0.0, value=21.0, step=0.1,
                                   help="Corporate tax rate applicable to the firm.")
        country_risk_premium = st.number_input("Country Risk Premium (%) :", min_value=0.0, value=0.2, step=0.01,
                                               help="Additional premium for country-specific risk.")
        specific_risk_premium = st.number_input("Specific Risk Premium (%) :", min_value=0.0, value=0.35, step=0.01,
                                                help="Additional premium for firm-specific risks.")
        size_premium = st.number_input("Size Premium (%) :", min_value=0.0, value=0.3, step=0.01,
                                       help="Premium for smaller firms.")
        submit = st.form_submit_button("Calculate WACC")

    if submit:
        try:
            financials_period = handler.get_financials_for_period(latest_period)
            balance_sheet = financials_period['balance_sheet']
            cashflow = financials_period['cashflow']
            keystats = financials_period['keystats']
            income_statement = financials_period['income_statement']

            if balance_sheet.empty or cashflow.empty or income_statement.empty:
                st.error(f"❌ Incomplete data for period {latest_period}. Ensure all statements are provided.")
                return

            # Calculate Beta
            beta = keystats.get('Beta', None)
            if pd.isna(beta) or beta == 0:
                # Attempt to calculate Beta from historical data
                beta_calculated = handler.calculate_beta()
                if beta_calculated is not None:
                    beta = beta_calculated
                    st.info(f"✅ Beta calculated from historical data: {beta:.4f}")
                else:
                    st.error("❌ Failed to determine Beta. Provide Beta in Key Stats or ensure historical data is complete.")
                    return
            else:
                beta = float(beta) if pd.notnull(beta) else None
                if beta is None:
                    st.error("❌ Invalid Beta value in Key Stats.")
                    return

            # Identify if company is a bank
            bank_indicators = [
                'Customers Deposits',
                'Credit Facilities',
                'Credit Facilities to Governmental Entities',
                'Due to Banks and Credit Institutions',
                'Term Deposits'
            ]
            is_bank = any(indicator in balance_sheet.index for indicator in bank_indicators)
            interest_paid = balance_sheet.get('Interest Paid for Borrowing', 0)
            is_islamic = is_bank and (interest_paid == 0)

            # Calculate Total Debt
            if is_bank:
                if is_islamic:
                    debt_fields = [
                        'Debt securities',
                        'Customers Deposits',
                        'Term Deposits',
                        'Due to Banks and Credit Institutions',
                        'Credit Facilities',
                        'Credit Facilities to Governmental Entities'
                    ]
                    existing_debt_fields = [field for field in debt_fields if field in balance_sheet.index]
                    total_debt = balance_sheet[existing_debt_fields].sum()
                else:
                    total_debt = balance_sheet.get('Total Liabilities', 0)
            else:
                current_portion = balance_sheet.get('Current Portion of Loan Payable', 0)
                long_term_debt = balance_sheet.get('Long Term Debt', 0)
                total_debt = current_portion + long_term_debt

            total_debt = float(total_debt or 0.0)
            total_equity = float(balance_sheet.get("Total Stockholders' Equity", 0) or 0.0)
            cash = float(balance_sheet.get("Cash", 0) or 0.0)

            invested_capital = total_debt + total_equity - cash
            if invested_capital <= 0:
                st.error("⚠️ Invested Capital is zero or negative. Cannot compute WACC.")
                return

            # Cost of Debt
            interest_expense = float(income_statement.get('Interest Expense', 0) or 0.0)
            cost_of_debt = abs(interest_expense) / total_debt if total_debt != 0 else 0
            cost_of_debt_after_tax = cost_of_debt * (1 - tax_rate / 100)

            # Cost of Equity (CAPM + Adjustments)
            equity_risk_premium = (market_return_input - risk_free_rate_input) / 100
            cost_of_equity = ((risk_free_rate_input / 100) +
                               (beta * equity_risk_premium) +
                               (country_risk_premium / 100) +
                               (specific_risk_premium / 100) +
                               (size_premium / 100))

            # Market values (proxy: using historical prices)
            df_prices = financials.get('historical_prices', pd.DataFrame())
            latest_price = df_prices.iloc[-1]['Close'] if not df_prices.empty else 0.0

            # Use the outstanding_shares from session state (user input)
            market_value_of_equity = (latest_price * st.session_state["outstanding_shares"]) / 1e6
            total_firm_value = market_value_of_equity + total_debt
            equity_weight = (market_value_of_equity / total_firm_value) if total_firm_value != 0 else 0
            debt_weight = (total_debt / total_firm_value) if total_firm_value != 0 else 0

            # Current WACC
            wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt_after_tax)

            # Calculate Target Debt-to-Equity Ratio from data instead of user input
            if total_equity != 0:
                target_debt_equity_ratio = total_debt / total_equity
            else:
                target_debt_equity_ratio = 0.0

            # Unlevered Beta & Relevered Beta (Hamada equation)
            if total_equity != 0:
                unlevered_beta = beta / (1 + ((1 - tax_rate / 100) * (total_debt / total_equity)))
            else:
                unlevered_beta = 0

            # Re-lever at target D/E
            target_equity_weight = 1 / (1 + target_debt_equity_ratio)
            target_debt_weight = target_debt_equity_ratio / (1 + target_debt_equity_ratio)
            relevered_beta_target = unlevered_beta * (1 + ((1 - tax_rate / 100) * target_debt_equity_ratio))
            cost_of_equity_target = ((risk_free_rate_input / 100) +
                                     (relevered_beta_target * equity_risk_premium) +
                                     (country_risk_premium / 100) +
                                     (specific_risk_premium / 100) +
                                     (size_premium / 100))
            wacc_target = (target_equity_weight * cost_of_equity_target) + (target_debt_weight * cost_of_debt_after_tax)

            # Book WACC could be similarly calculated using book values
            wacc_book = wacc

            # Display Results
            st.subheader(f"📊 WACC: {format_percentage(wacc * 100)}")
            st.subheader(f"🎯 Target WACC: {format_percentage(wacc_target * 100)}")
            st.subheader(f"📚 Book WACC: {format_percentage(wacc_book * 100)}")

            with st.expander("🔍 Detailed Parameters"):
                details = {
                    "Risk-Free Rate (Rf)": format_percentage(risk_free_rate_input),
                    "Market Return (Rm)": format_percentage(market_return_input),
                    "Beta": f"{beta:.4f}",
                    "Equity Risk Premium (ERP)": format_percentage(equity_risk_premium * 100),
                    "Country Risk Premium": format_percentage(country_risk_premium),
                    "Specific Risk Premium": format_percentage(specific_risk_premium),
                    "Size Premium": format_percentage(size_premium),
                    "Unlevered Beta": f"{unlevered_beta:.4f}",
                    "Relevered Beta (Target)": f"{relevered_beta_target:.4f}",
                    "Cost of Equity (Current)": format_percentage(cost_of_equity * 100),
                    "Cost of Equity (Target)": format_percentage(cost_of_equity_target * 100),
                    "Cost of Debt After Tax": format_percentage(cost_of_debt_after_tax * 100),
                    "Equity Weight (Current)": f"{equity_weight:.4f}",
                    "Debt Weight (Current)": f"{debt_weight:.4f}",
                    "Current D/E Ratio": f"{total_debt/total_equity:.4f}" if total_equity != 0 else "N/A",
                    "Target D/E Ratio": f"{target_debt_equity_ratio:.4f}",
                    "WACC (%)": format_percentage(wacc * 100),
                    "WACC Target (%)": format_percentage(wacc_target * 100),
                    "WACC Book (%)": format_percentage(wacc_book * 100)
                }
                st.table(pd.DataFrame(list(details.items()), columns=["Parameter", "Value"]))

            # Sensitivity Analysis
            st.markdown("### 📊 Sensitivity Analysis")
            sensitivity_data = []
            cost_of_equity_variations = [cost_of_equity * 0.9, cost_of_equity, cost_of_equity * 1.1]
            cost_of_debt_variations = [cost_of_debt_after_tax * 0.9, cost_of_debt_after_tax, cost_of_debt_after_tax * 1.1]
            for coe in cost_of_equity_variations:
                for cod in cost_of_debt_variations:
                    wacc_calculated = (equity_weight * coe) + (debt_weight * cod)
                    sensitivity_data.append({
                        "Cost of Equity (%)": f"{coe * 100:.2f}",
                        "Cost of Debt After Tax (%)": f"{cod * 100:.2f}",
                        "WACC (%)": wacc_calculated * 100
                    })

            sensitivity_df = pd.DataFrame(sensitivity_data)
            if not sensitivity_df.empty:
                st.dataframe(sensitivity_df)
                try:
                    pivot_table = sensitivity_df.pivot(index="Cost of Equity (%)", columns="Cost of Debt After Tax (%)", values="WACC (%)")
                    fig = px.imshow(
                        pivot_table.astype(float),
                        labels=dict(x="Cost of Debt After Tax (%)", y="Cost of Equity (%)", color="WACC (%)"),
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        color_continuous_scale='Viridis',
                        title='📊 Sensitivity Analysis Heatmap'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Unable to generate heatmap: {e}")
                    logging.error(f"Sensitivity Analysis Heatmap Error: {e}")
            else:
                st.info("⚠️ Not enough data for sensitivity scenarios.")

            # WACC Components Visualization
            wacc_components = pd.DataFrame({
                "Component": ["Equity", "Debt"],
                "Weight": [equity_weight, debt_weight],
                "Cost (%)": [cost_of_equity * 100, cost_of_debt_after_tax * 100]
            })
            fig = px.bar(wacc_components, x='Component', y='Weight', title='📊 WACC Components by Weight', text='Weight')
            st.plotly_chart(fig, use_container_width=True)

            # Export WACC Results
            wacc_export_df = pd.DataFrame({
                "WACC Type": ["Current WACC", "Target WACC", "Book WACC"],
                "WACC (%)": [wacc * 100, wacc_target * 100, wacc_book * 100]
            })
            export_results_as_csv(wacc_export_df, 'WACC_Results.csv', "Download WACC Results as CSV")

        except Exception as e:
            st.error(f"❌ Error in WACC calculation: {e}")
            logging.error(f"WACC Calculator Error: {e}")
    st.markdown("---")


def dcf_calculator(financials: Dict[str, Any]):
    st.header("🔮 DCF Calculator")
    st.markdown("""
    **DCF (Discounted Cash Flow)** valuation derived from historical data and key stats, with user option to override WACC.
    
    Steps:
    - We estimate starting FCF, growth, terminal growth, and exit multiple from data.
    - We give the user the option to input their own WACC (discount rate).
    - Show three valuations: Optimistic, Base, and Pessimistic scenarios.
    """)

    handler = FinancialDataHandler(financials)
    latest_period = handler.get_latest_financial_period()
    if not latest_period:
        st.error("❌ No financial period data available.")
        return

    financials_period = handler.get_financials_for_period(latest_period)
    cashflow = financials_period['cashflow']
    keystats = financials_period['keystats']

    if cashflow.empty:
        st.error(f"❌ Cash Flow data for period {latest_period} is missing.")
        return

    # -------------------------
    # Derive Starting FCF, Growth, Terminal Growth, Exit Multiple from data
    # -------------------------
    historical_periods = handler.periods[:5]
    historical_fcfs = []
    for period in historical_periods:
        if period in cashflow.index:
            period_ocf = cashflow.loc[period, 'Net Cash from Operating Activities'] if 'Net Cash from Operating Activities' in cashflow.columns else 0
            period_capex = cashflow.loc[period, 'Purchase of Property, Plant & Equipment'] if 'Purchase of Property, Plant & Equipment' in cashflow.columns else 0
            try:
                period_ocf = float(period_ocf or 0)
                period_capex = float(period_capex or 0)
            except ValueError:
                continue
            hist_fcf = period_ocf - abs(period_capex)
            historical_fcfs.append((period, hist_fcf))

    positive_fcfs = [fcf for (p, fcf) in historical_fcfs if fcf > 0]
    if positive_fcfs:
        latest_pos_fcf = positive_fcfs[0]
        starting_fcf = latest_pos_fcf
    else:
        if historical_fcfs:
            avg_fcf = sum(f for (p, f) in historical_fcfs) / len(historical_fcfs)
            starting_fcf = avg_fcf if avg_fcf > 0 else 1000.0
        else:
            starting_fcf = 1000.0

    def parse_period_dt(p_str):
        date_str = p_str.split('ending')[-1].strip()
        return datetime.strptime(date_str, '%Y-%m')

    positive_fcfs_sorted = sorted([(p, f) for (p, f) in historical_fcfs if f > 0], key=lambda x: parse_period_dt(x[0]))

    if len(positive_fcfs_sorted) >= 2:
        start_val = positive_fcfs_sorted[0][1]
        end_val = positive_fcfs_sorted[-1][1]
        periods_count = len(positive_fcfs_sorted) - 1
        if start_val > 0:
            historical_cagr = (end_val / start_val)**(1 / periods_count) - 1
        else:
            historical_cagr = 0.05
    else:
        historical_cagr = 0.05

    terminal_growth_rate = min(historical_cagr / 2, 0.02)
    if terminal_growth_rate < 0:
        terminal_growth_rate = 0.02

    wacc_market = keystats.get('WACC Market', 10.0)
    try:
        if isinstance(wacc_market, str):
            wacc_market = float(wacc_market.replace('%', ''))
        if wacc_market > 1:
            wacc_market /= 100.0
    except ValueError:
        wacc_market = 0.10
    if not (0.01 <= wacc_market <= 0.5):
        wacc_market = 0.10

    exit_multiple = keystats.get('Exit Multiple', 10.0)
    try:
        exit_multiple = float(exit_multiple)
        if exit_multiple <= 0:
            exit_multiple = 10.0
    except:
        exit_multiple = 10.0

    projection_years = 5

    # -------------------------
    # Let user input WACC
    # -------------------------
    with st.form("dcf_wacc_form"):
        st.subheader("📋 Input Parameters (Optional)")
        user_wacc_input = st.number_input("Enter Your WACC (%) [0-100]:", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                          help="If you enter a value > 0, this will override the calculated WACC.")
        submit_wacc = st.form_submit_button("Apply WACC")

    if submit_wacc:
        if user_wacc_input > 0 and user_wacc_input <= 100:
            discount_rate = user_wacc_input / 100.0
            st.info(f"Using user provided WACC: {user_wacc_input:.2f}%")
        else:
            discount_rate = wacc_market
            st.warning("Invalid WACC entered. Using derived WACC from data.")
    else:
        discount_rate = wacc_market

    def project_dcf(fcf_start, growth_rate, discount_rate, tgr, multiple):
        projected_cf = []
        current_fcf = fcf_start
        for year in range(1, projection_years + 1):
            current_fcf *= (1 + growth_rate)
            projected_cf.append((year, current_fcf))

        pv_sum = 0
        dcf_rows = []
        for (yr, cf_val) in projected_cf:
            df = (1 + discount_rate)**yr
            pv = cf_val / df
            pv_sum += pv
            dcf_rows.append({
                "Year": yr,
                "Projected FCF": cf_val,
                "Discount Factor": df,
                "Present Value": pv
            })

        terminal_fcf = projected_cf[-1][1]
        terminal_value = terminal_fcf * multiple
        tv_pv = terminal_value / ((1 + discount_rate)**projection_years)
        total_value = pv_sum + tv_pv

        return total_value, dcf_rows, terminal_value, tv_pv

    # Scenarios:
    base_growth = historical_cagr
    base_discount = discount_rate
    base_tgr = terminal_growth_rate
    base_multiple = exit_multiple
    base_value, base_rows, base_tv, base_tv_pv = project_dcf(starting_fcf, base_growth, base_discount, base_tgr, base_multiple)

    # Optimistic scenario
    optimistic_growth = base_growth * 1.1
    optimistic_discount = base_discount * 0.9
    optimistic_multiple = base_multiple * 1.1
    optimistic_value, _, _, _ = project_dcf(starting_fcf, optimistic_growth, optimistic_discount, base_tgr, optimistic_multiple)

    # Pessimistic scenario
    pessimistic_growth = base_growth * 0.9
    pessimistic_discount = base_discount * 1.1
    pessimistic_multiple = base_multiple * 0.9
    pessimistic_value, _, _, _ = project_dcf(starting_fcf, pessimistic_growth, pessimistic_discount, base_tgr, pessimistic_multiple)

    currency_symbol = {
        "IRR": "IRR ",
        "USD": "$",
        "EUR": "€",
        "GBP": "£"
    }.get(st.session_state.get('dcf_currency', 'IRR'), 'IRR ')

    st.subheader("Enterprise Value (Base Scenario)")
    st.markdown(f"**Enterprise Value (DCF):** {format_number(base_value, currency_symbol)}")

    dcf_df = pd.DataFrame(base_rows)
    dcf_df['Projected FCF'] = dcf_df['Projected FCF'].apply(lambda x: format_number(x, currency_symbol))
    dcf_df['Present Value'] = dcf_df['Present Value'].apply(lambda x: format_number(x, currency_symbol))
    dcf_df['Discount Factor'] = dcf_df['Discount Factor'].apply(lambda x: f"{x:.4f}")
    st.table(dcf_df)

    st.markdown(f"**Terminal Value:** {format_number(base_tv, currency_symbol)}")
    st.markdown(f"**Terminal Value (Present Value):** {format_number(base_tv_pv, currency_symbol)}")
    st.markdown(f"**Total Enterprise Value:** {format_number(base_value, currency_symbol)}")

    # Show scenarios
    st.markdown("### Scenario Analysis")
    scenario_data = {
        "Scenario": ["Pessimistic", "Base", "Optimistic"],
        "Growth Rate (%)": [pessimistic_growth*100, base_growth*100, optimistic_growth*100],
        "Discount Rate (%)": [pessimistic_discount*100, base_discount*100, optimistic_discount*100],
        "Exit Multiple": [pessimistic_multiple, base_multiple, optimistic_multiple],
        f"Enterprise Value ({currency_symbol})": [pessimistic_value, base_value, optimistic_value]
    }
    scenario_df = pd.DataFrame(scenario_data)
    scenario_df["Growth Rate (%)"] = scenario_df["Growth Rate (%)"].apply(lambda x: format_percentage(x))
    scenario_df["Discount Rate (%)"] = scenario_df["Discount Rate (%)"].apply(lambda x: format_percentage(x))
    scenario_df["Exit Multiple"] = scenario_df["Exit Multiple"].apply(lambda x: f"{x:.2f}x")
    scenario_df[f"Enterprise Value ({currency_symbol})"] = scenario_df[f"Enterprise Value ({currency_symbol})"].apply(lambda x: format_number(x, currency_symbol))

    st.table(scenario_df)

    # Export Results
    export_results_as_csv(pd.DataFrame(base_rows), 'DCF_Table.csv', "Download DCF Table as CSV")
    export_results_as_csv(pd.DataFrame(scenario_data), 'Scenario_Analysis.csv', "Download Scenario Analysis as CSV")

    st.markdown("---")
    st.markdown("**Note:** The above values are derived from historical data and user WACC input. Adjust assumptions as needed.")

# -----------------------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------------------
@st.cache_data
def load_excel_files(balance_sheet_file, cash_flow_file, income_statement_file, keystats_file, historical_prices_file, market_index_file, risk_free_rate_file) -> Dict[str, pd.DataFrame]:
    """
    Load and preprocess financial data from uploaded Excel files.
    Ensures consistent formats and numeric conversions.
    """
    try:
        financial_data = {}

        if balance_sheet_file is not None:
            bs_df = pd.read_excel(balance_sheet_file, sheet_name=0)
            financial_data['balance_sheet'] = preprocess_balance_sheet(bs_df)
        else:
            st.error("❌ Balance Sheet file is missing.")
            return {}

        if cash_flow_file is not None:
            cf_df = pd.read_excel(cash_flow_file, sheet_name=0)
            financial_data['cashflow'] = preprocess_cash_flow(cf_df)
        else:
            st.error("❌ Cash Flow Statement file is missing.")
            return {}

        if income_statement_file is not None:
            is_df = pd.read_excel(income_statement_file, sheet_name=0)
            financial_data['income_statement'] = preprocess_income_statement(is_df)
        else:
            st.error("❌ Income Statement file is missing.")
            return {}

        if keystats_file is not None:
            ks_df = pd.read_excel(keystats_file, sheet_name=0)
            financial_data['keystats'] = preprocess_keystats(ks_df)
        else:
            st.error("❌ Key Statistics file is missing.")
            return {}

        if historical_prices_file is not None:
            hp_df = pd.read_excel(historical_prices_file, sheet_name=0)
            financial_data['historical_prices'] = preprocess_historical_prices(hp_df)
        else:
            st.error("❌ Historical Prices file is missing.")
            return {}

        if market_index_file is not None:
            mi_df = pd.read_excel(market_index_file, sheet_name=0)
            financial_data['market_index'] = preprocess_market_index(mi_df)
        else:
            st.error("❌ Market Index file is missing.")
            return {}

        if risk_free_rate_file is not None:
            rf_df = pd.read_excel(risk_free_rate_file, sheet_name=0)
            financial_data['risk_free_rate'] = preprocess_risk_free_rate(rf_df)
        else:
            st.error("❌ Risk-Free Rate file is missing.")
            return {}

        return financial_data

    except Exception as e:
        st.error(f"❌ Error loading Excel files: {e}")
        logging.error(f"Excel Loading Error: {e}")
        return {}

def preprocess_balance_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Balance Sheet data: transpose, convert to numeric."""
    try:
        df = df.set_index(df.columns[0]).transpose()
        df.index = df.index.map(str)
        df.replace('-', 0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing Balance Sheet: {e}")
        logging.error(f"Balance Sheet Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_cash_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Cash Flow Statement data."""
    try:
        df = df.set_index(df.columns[0]).transpose()
        df.index = df.index.map(str)
        df.replace('-', 0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing Cash Flow Statement: {e}")
        logging.error(f"Cash Flow Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_income_statement(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Income Statement data."""
    try:
        df = df.set_index(df.columns[0]).transpose()
        df.index = df.index.map(str)
        df.replace('-', 0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing Income Statement: {e}")
        logging.error(f"Income Statement Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_keystats(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Key Statistics data."""
    try:
        df = df.set_index(df.columns[0]).transpose()
        df.index = df.index.map(str)
        df.replace('-', 0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing Key Statistics: {e}")
        logging.error(f"Key Statistics Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_historical_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Historical Prices data."""
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'VWap', 'Change', '%', 'Market Cap',
                         'Trades #', 'Volume', 'Value', 'Free Float', 'P/E (ttm)', 'P/B', 'P/S',
                         'P/NAV', 'NAV', 'Individual Buy Power', 'Relative Net Individual',
                         'Net Individual', 'Per Individual Purchases', 'Per Individual Sales']
        existing_price_columns = [col for col in price_columns if col in df.columns]
        df[existing_price_columns] = df[existing_price_columns].replace('-', 0)
        df[existing_price_columns] = df[existing_price_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing Historical Prices: {e}")
        logging.error(f"Historical Prices Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_market_index(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Market Index data."""
    try:
        df = df.rename(columns={
            'تاریخ میلادی': 'Date_Gregorian',
            'تاریخ شمسی': 'Date_Shamsi',
            'مقدار': 'Market_Index'
        })
        df['Date_Gregorian'] = pd.to_datetime(df['Date_Gregorian'], errors='coerce')
        df.dropna(subset=['Date_Gregorian'], inplace=True)
        df.sort_values('Date_Gregorian', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['Market_Index'] = pd.to_numeric(df['Market_Index'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing Market Index: {e}")
        logging.error(f"Market Index Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_risk_free_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Risk-Free Rate data."""
    try:
        df = df.rename(columns={
            'تاریخ میلادی': 'Date_Gregorian',
            'تاریخ شمسی': 'Date_Shamsi',
            'مقدار': 'Risk_Free_Rate'
        })
        df['Date_Gregorian'] = pd.to_datetime(df['Date_Gregorian'], errors='coerce')
        df.dropna(subset=['Date_Gregorian'], inplace=True)
        df.sort_values('Date_Gregorian', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['Risk_Free_Rate'] = pd.to_numeric(df['Risk_Free_Rate'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ Error preprocessing Risk-Free Rate: {e}")
        logging.error(f"Risk-Free Rate Preprocessing Error: {e}")
        return pd.DataFrame()

def main():
    """
    Main function to run the Streamlit application.
    Users can upload financial data and access ROIC, WACC, and DCF calculators.
    """
    st.title("💡 Financial Calculators")

    st.markdown("""
    Welcome to the **Financial Calculators** app! 
    \n**Instructions:**
    1. Upload all required Excel files (Balance Sheet, Cash Flow, Income Statement, Key Stats, Historical Prices, Market Index, Risk-Free Rate) in the sidebar.
    2. Click "Load Financial Data".
    3. Select a calculator (ROIC, WACC, or DCF) from the sidebar.
    4. Adjust parameters and run calculations.
    
    **Disclaimer:** 
    The results are for educational and informational purposes only and are based on provided data and standard finance methods. 
    Please consult a financial professional before making any investment decisions.
    """)

    # File Uploads
    st.sidebar.header("📂 Upload Financial Data")
    balance_sheet_file = st.sidebar.file_uploader("Balance Sheet Excel File", type=["xlsx", "xls"])
    cash_flow_file = st.sidebar.file_uploader("Cash Flow Excel File", type=["xlsx", "xls"])
    income_statement_file = st.sidebar.file_uploader("Income Statement Excel File", type=["xlsx", "xls"])
    keystats_file = st.sidebar.file_uploader("Key Stats Excel File", type=["xlsx", "xls"])
    historical_prices_file = st.sidebar.file_uploader("Historical Prices Excel File", type=["xlsx", "xls"])
    market_index_file = st.sidebar.file_uploader("Market Index Excel File", type=["xlsx", "xls"])
    risk_free_rate_file = st.sidebar.file_uploader("Risk-Free Rate Excel File", type=["xlsx", "xls"])

    if st.sidebar.button("Load Financial Data"):
        if not (balance_sheet_file and cash_flow_file and income_statement_file and keystats_file and historical_prices_file and market_index_file and risk_free_rate_file):
            st.sidebar.error("❌ Please upload all required Excel files before loading.")
            st.stop()

        financial_data = load_excel_files(balance_sheet_file, cash_flow_file, income_statement_file, keystats_file, historical_prices_file, market_index_file, risk_free_rate_file)
        if financial_data:
            st.session_state['financial_data'] = financial_data
            st.sidebar.success("✅ Financial data loaded successfully.")
        else:
            st.sidebar.error("❌ Failed to load financial data.")
            st.stop()

    if 'financial_data' not in st.session_state:
        st.info("🔍 Please upload and load all required Excel files from the sidebar to access the calculators.")
        st.stop()

    financials = st.session_state['financial_data']

    # Sidebar for Calculator Selection
    st.sidebar.header("🛠️ Select Calculator")
    calculator_tabs = ["ROIC", "WACC", "DCF"]
    selected_tab = st.sidebar.radio("Choose a Calculator:", options=calculator_tabs)

    st.markdown("---")

    with st.container():
        if selected_tab == "ROIC":
            roic_calculator(financials)
        elif selected_tab == "WACC":
            wacc_calculator(financials)
        elif selected_tab == "DCF":
            dcf_calculator(financials)
        else:
            st.error("❌ Invalid calculator selection.")

    st.markdown("""
    ---
    © 2024 Financial Calculators App by Navid Ramezani. All rights reserved.
    """)

if __name__ == "__main__":
    main()
