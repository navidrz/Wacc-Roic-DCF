import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Any, Dict, Optional
import logging
from datetime import datetime

# -----------------------------------------------------------
# پیکربندی صفحه، فونت وزیر، راست‌چین کردن و معرفی ابزار
# -----------------------------------------------------------
st.set_page_config(
    page_title="💡 ماشین حساب مالی",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': (
            "💡 **محاسبه‌گرهای مالی**\n\n"
            "با این ابزار می‌توانید معیارهای کلیدی مالی مانند:\n"
            "- **ROIC (بازده سرمایه)**\n"
            "- **WACC (میانگین موزون هزینه سرمایه)**\n"
            "- **DCF (جریان نقدی تنزیل‌شده)**\n\n"
            "را محاسبه کنید. تمامی محاسبات بر اساس مراجع معتبر مالی مانند:\n"
            "- *Brealey, Myers & Allen - Principles of Corporate Finance*\n"
            "- *McKinsey & Company - Valuation*\n\n"
            "این ابزار فقط برای اهداف آموزشی و تحلیلی ارائه شده است."
        )
    }
)

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font@v30.1.0/dist/font-face.css');
html, body, [class*="css"] {
    font-family: Vazir, sans-serif;
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# انیمیشن‌ها و لوگو
# -----------------------------------------------------------
# گیف‌های جدید برای زیبایی بیشتر
gif_roic = "https://media.giphy.com/media/3oEdva9BUHp5mtykXu/giphy.gif"   # گربه با ماشین حساب
gif_wacc = "https://media.giphy.com/media/l0MYClFLwRE4fOAfK/giphy.gif"   # مردی که مشغول حساب و کتاب است
gif_dcf = "https://media.giphy.com/media/26xBzgcrhAXi9BMuk/giphy.gif"    # بارش پول
gif_main = "https://media.giphy.com/media/3orieT29z6jgK6A1s0/giphy.gif"  # گربه با عینک آفتابی و پول
logo_url = "https://iranbourse.net/wp-content/uploads/2023/01/iranbourse-logo.png"

# -----------------------------------------------------------
# تنظیمات لاگ
# -----------------------------------------------------------
logging.basicConfig(
    filename='financial_calculators.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# -----------------------------------------------------------
# توابع کمکی
# -----------------------------------------------------------
def format_number(num: float, currency_symbol: str = "") -> str:
    try:
        return f"{currency_symbol}{num:,.2f}"
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(num: float) -> str:
    try:
        return f"{num:.2f}%"
    except (ValueError, TypeError):
        return "N/A"

def export_results_as_csv(dataframe: pd.DataFrame, filename: str, label: str):
    try:
        csv = dataframe.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"❌ خطا در خروجی گرفتن فایل CSV: {e}")
        logging.error(f"CSV Export Error: {e}")

# -----------------------------------------------------------
# کلاس مدیریت داده‌های مالی
# -----------------------------------------------------------
class FinancialDataHandler:
    def __init__(self, financial_data: Dict[str, pd.DataFrame]):
        self.financial_data = financial_data
        self.periods = self.get_all_periods()

    def get_all_periods(self) -> list:
        try:
            bs_periods = list(self.financial_data['balance_sheet'].index)
            sorted_periods = sorted(bs_periods, key=lambda x: self.parse_period(x), reverse=True)
            return sorted_periods
        except KeyError as e:
            logging.error(f"Missing financial statement for periods: {e}")
            return []

    def parse_period(self, period_str: str) -> datetime:
        try:
            date_str = period_str.split('ending')[-1].strip()
            return datetime.strptime(date_str, '%Y-%m')
        except Exception as e:
            logging.error(f"Error parsing period '{period_str}': {e}")
            return datetime.min

    def get_latest_financial_period(self) -> Optional[str]:
        return self.periods[0] if self.periods else None

    def get_financials_for_period(self, period: str) -> Dict[str, pd.Series]:
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
        try:
            df_stock = self.financial_data.get('historical_prices', pd.DataFrame())
            df_market = self.financial_data.get('market_index', pd.DataFrame())
            df_risk_free = self.financial_data.get('risk_free_rate', pd.DataFrame())

            if df_stock.empty or df_market.empty or df_risk_free.empty:
                st.warning("⚠️ داده‌های کافی برای محاسبه بتا وجود ندارد.")
                return None

            df_market = df_market.rename(columns={'تاریخ میلادی': 'Date_Gregorian', 'مقدار': 'Market_Index'})
            df_risk_free = df_risk_free.rename(columns={'date': 'Date_Gregorian', 'ytm': 'Risk_Free_Rate'})

            df = pd.merge(df_stock[['Date', 'Close']], df_market[['Date_Gregorian', 'Market_Index']],
                          left_on='Date', right_on='Date_Gregorian', how='inner')
            df = pd.merge(df, df_risk_free[['Date_Gregorian', 'Risk_Free_Rate']], on='Date_Gregorian', how='inner')

            if df.empty:
                st.warning("⚠️ تاریخ‌های مشترک بین قیمت سهم، شاخص بازار و نرخ بدون ریسک یافت نشد.")
                return None

            df.sort_values('Date', inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce').fillna(0)
            df['Market_Index'] = pd.to_numeric(df['Market_Index'], errors='coerce').fillna(0)
            df['Risk_Free_Rate'] = pd.to_numeric(df['Risk_Free_Rate'], errors='coerce').fillna(0)
            df['Stock_Return'] = df['Close'].pct_change()
            df['Market_Return'] = df['Market_Index'].pct_change()
            df['Risk_Free_Return'] = df['Risk_Free_Rate'] / 100
            df.dropna(inplace=True)

            if df.empty:
                st.warning("⚠️ پس از محاسبه بازده‌ها داده کافی وجود ندارد.")
                return None

            df['Excess_Stock_Return'] = df['Stock_Return'] - df['Risk_Free_Return']
            df['Excess_Market_Return'] = df['Market_Return'] - df['Risk_Free_Return']

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=['Excess_Stock_Return', 'Excess_Market_Return'], inplace=True)

            if df['Excess_Market_Return'].nunique() <= 1 or df['Excess_Stock_Return'].nunique() <= 1:
                st.error("❌ تنوع کافی در داده‌ها برای محاسبه بتا وجود ندارد.")
                return None

            st.markdown("### 📊 داده‌های پردازش‌شده برای محاسبه بتا")
            st.dataframe(df[['Date', 'Excess_Market_Return', 'Excess_Stock_Return']].head())

            X = df['Excess_Market_Return'].values.reshape(-1, 1)
            y = df['Excess_Stock_Return'].values
            reg = LinearRegression()
            reg.fit(X, y)
            beta = reg.coef_[0]
            r_squared = reg.score(X, y)

            fig = px.scatter(
                x=df['Excess_Market_Return'], y=df['Excess_Stock_Return'],
                labels={'x': 'بازده مازاد بازار', 'y': 'بازده مازاد سهم'},
                title='📈 رابطه بازده مازاد سهم و بازار'
            )
            x_range = np.linspace(df['Excess_Market_Return'].min(), df['Excess_Market_Return'].max(), 100)
            y_pred = reg.predict(x_range.reshape(-1, 1))
            fig.add_traces(px.line(x=x_range, y=y_pred).data)

            fig.add_annotation(
                x=0.05, y=0.95, xref="paper", yref="paper",
                text=f"Beta: {beta:.4f}<br>R²: {r_squared:.4f}",
                showarrow=False, align='left', bgcolor="rgba(0,0,0,0.5)",
                font=dict(color="white")
            )
            st.plotly_chart(fig, use_container_width=True)

            return beta
        except Exception as e:
            st.error(f"❌ خطا در محاسبه بتا: {e}")
            logging.error(f"Beta Calculation Error: {e}")
            return None

# -----------------------------------------------------------
# توضیحات صفحات
# -----------------------------------------------------------
def show_intro_page():
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:20px;">
        <a href="http://www.iranbourse.net" target="_blank">
            <img src="{logo_url}" alt="Iranbours Logo" width="150" style="cursor:pointer;"/>
        </a>
    </div>

    <div style="text-align: center;">
        <img src="{gif_main}" width="150" />
    </div>
    
    ### درباره ابزار محاسبه‌گرهای مالی
    
    این ابزار، با الهام از اصول مالی بین‌المللی، بستری فراهم می‌کند تا با بارگذاری فایل‌های اکسل حاوی داده‌های مالی، به آسانی شاخص‌های مهمی مثل **ROIC**، **WACC** و **DCF** را محاسبه کنید. طراحی این سامانه به گونه‌ای است که چه تحلیل‌گر حرفه‌ای باشید و چه دانشجوی تازه‌کار، بتوانید اطلاعات را به سرعت پردازش کرده و از نتایج در تصمیمات مالی خود بهره ببرید.
    
    تمام بخش‌ها با ظاهری کاربرپسند و سرعت بالا ارائه شده‌اند. دیگر نیازی به استفاده از نرم‌افزارهای پیچیده نیست. اینجا می‌توانید با چند کلیک ساده، اعداد، نمودارها و جداولی را مشاهده کنید که شناخت دقیق‌تری از وضعیت مالی شرکت‌ها به شما می‌دهد.  
    
    برای مطالعه مطالب آموزشی و مقالات تحلیلی بیشتر، به <a href="http://www.iranbourse.net" target="_blank">ایران بورس</a> سر بزنید.
    """, unsafe_allow_html=True)

def show_roic_page():
    st.markdown(f"""
    <div style="text-align: center;">
        <img src="{gif_roic}" width="150" />
    </div>
    
    ### محاسبه بازده سرمایه (ROIC)

    **بازده سرمایه (ROIC)** سنجه‌ای برای اندازه‌گیری کارایی شرکت در استفاده از منابع مالی خود است. این شاخص نشان می‌دهد تا چه حد سرمایه‌گذاری در یک شرکت به خلق سود عملیاتی پس از مالیات (NOPAT) منجر می‌شود. با فرمول **ROIC = NOPAT / Invested Capital**، می‌توانید ارزیابی کنید که مدیریت شرکت تا چه اندازه از سرمایه در دسترس استفاده بهینه می‌کند.
    """, unsafe_allow_html=True)

def show_wacc_page():
    st.markdown(f"""
    <div style="text-align: center;">
        <img src="{gif_wacc}" width="150" />
    </div>
    
    ### محاسبه میانگین موزون هزینه سرمایه (WACC)
    
    **WACC** نرخ بازده مورد انتظار کل سرمایه‌گذاران (سهامداران و وام‌دهندگان) را نشان می‌دهد. با استفاده از نرخ بدون ریسک، صرف ریسک بازار، بتا، و نرخ مالیات، می‌توانید WACC را محاسبه و در تصمیمات مربوط به ساختار سرمایه، ارزیابی پروژه‌ها و ارزش‌گذاری شرکت‌ها به کار گیرید.
    """, unsafe_allow_html=True)

def show_dcf_page():
    st.markdown(f"""
    <div style="text-align: center;">
        <img src="{gif_dcf}" width="150" />
    </div>
    
    ### محاسبه جریان نقدی تنزیل‌شده (DCF)
    
    **DCF** ابزاری قدرتمند برای برآورد ارزش ذاتی شرکت است. با تنزیل جریان‌های نقدی آتی بر مبنای نرخ تنزیل (WACC یا نرخ بازده مورد انتظار)، می‌توانید ارزش فعلی کسب‌وکار را محاسبه کرده و در سناریوهای مختلف (خوش‌بینانه، بدبینانه، پایه) نتایج را بررسی کنید.
    """, unsafe_allow_html=True)

# -----------------------------------------------------------
# توابع پیش‌پردازش داده‌ها
# -----------------------------------------------------------
def preprocess_balance_sheet(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.set_index(df.columns[0]).transpose()
        df.index = df.index.map(str)
        df.replace('-', 0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ خطا در پیش‌پردازش ترازنامه: {e}")
        logging.error(f"Balance Sheet Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_cash_flow(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.set_index(df.columns[0]).transpose()
        df.index = df.index.map(str)
        df.replace('-', 0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ خطا در پیش‌پردازش صورت جریان نقدینگی: {e}")
        logging.error(f"Cash Flow Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_income_statement(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.set_index(df.columns[0]).transpose()
        df.index = df.index.map(str)
        df.replace('-', 0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ خطا در پیش‌پردازش صورت سود و زیان: {e}")
        logging.error(f"Income Statement Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_keystats(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.set_index(df.columns[0]).transpose()
        df.index = df.index.map(str)
        df.replace('-', 0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"❌ خطا در پیش‌پردازش آمار کلیدی: {e}")
        logging.error(f"Key Statistics Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_historical_prices(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
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
        st.error(f"❌ خطا در پیش‌پردازش قیمت‌های تاریخی سهام: {e}")
        logging.error(f"Historical Prices Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_market_index(df: pd.DataFrame) -> pd.DataFrame:
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
        st.error(f"❌ خطا در پیش‌پردازش شاخص بازار: {e}")
        logging.error(f"Market Index Preprocessing Error: {e}")
        return pd.DataFrame()

def preprocess_risk_free_rate(df: pd.DataFrame) -> pd.DataFrame:
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
        st.error(f"❌ خطا در پیش‌پردازش نرخ بدون ریسک: {e}")
        logging.error(f"Risk-Free Rate Preprocessing Error: {e}")
        return pd.DataFrame()

@st.cache_data
def load_excel_files(balance_sheet_file, cash_flow_file, income_statement_file, keystats_file, historical_prices_file, market_index_file, risk_free_rate_file) -> Dict[str, pd.DataFrame]:
    try:
        financial_data = {}
        if balance_sheet_file is not None:
            bs_df = pd.read_excel(balance_sheet_file, sheet_name=0)
            financial_data['balance_sheet'] = preprocess_balance_sheet(bs_df)
        else:
            st.error("❌ فایل ترازنامه موجود نیست.")
            return {}

        if cash_flow_file is not None:
            cf_df = pd.read_excel(cash_flow_file, sheet_name=0)
            financial_data['cashflow'] = preprocess_cash_flow(cf_df)
        else:
            st.error("❌ فایل صورت جریان نقدینگی موجود نیست.")
            return {}

        if income_statement_file is not None:
            is_df = pd.read_excel(income_statement_file, sheet_name=0)
            financial_data['income_statement'] = preprocess_income_statement(is_df)
        else:
            st.error("❌ فایل صورت سود و زیان موجود نیست.")
            return {}

        if keystats_file is not None:
            ks_df = pd.read_excel(keystats_file, sheet_name=0)
            financial_data['keystats'] = preprocess_keystats(ks_df)
        else:
            st.error("❌ فایل آمار کلیدی موجود نیست.")
            return {}

        if historical_prices_file is not None:
            hp_df = pd.read_excel(historical_prices_file, sheet_name=0)
            financial_data['historical_prices'] = preprocess_historical_prices(hp_df)
        else:
            st.error("❌ فایل قیمت‌های تاریخی سهام موجود نیست.")
            return {}

        if market_index_file is not None:
            mi_df = pd.read_excel(market_index_file, sheet_name=0)
            financial_data['market_index'] = preprocess_market_index(mi_df)
        else:
            st.error("❌ فایل شاخص بازار موجود نیست.")
            return {}

        if risk_free_rate_file is not None:
            rf_df = pd.read_excel(risk_free_rate_file, sheet_name=0)
            financial_data['risk_free_rate'] = preprocess_risk_free_rate(rf_df)
        else:
            st.error("❌ فایل نرخ بدون ریسک موجود نیست.")
            return {}

        return financial_data

    except Exception as e:
        st.error(f"❌ خطا در بارگذاری فایل‌های اکسل: {e}")
        logging.error(f"Excel Loading Error: {e}")
        return {}

# -----------------------------------------------------------
# توابع محاسباتی ROIC، WACC، DCF
# -----------------------------------------------------------

def roic_calculator(financials: Dict[str, pd.DataFrame]):
    show_roic_page()
    st.markdown("""
    این ابزار با وارد کردن داده‌های مالی، ROIC را به صورت خودکار محاسبه می‌کند.
    """)
    handler = FinancialDataHandler(financials)
    latest_period = handler.get_latest_financial_period()

    if not latest_period:
        st.error("❌ داده‌های مالی کافی وجود ندارد. لطفاً داده‌ها را بارگذاری کنید.")
        return

    with st.form("roic_form"):
        st.subheader("📋 تنظیمات ورودی ROIC")
        tax_rate = st.number_input("نرخ مالیات (%) برای ROIC:", min_value=0.0, value=21.0, step=0.1,
                                   help="نرخ مالیات برای محاسبه NOPAT استفاده می‌شود.")
        period = st.selectbox("دوره مالی مورد نظر برای ROIC:", options=handler.periods,
                              help="دوره‌ای که می‌خواهید ROIC برای آن محاسبه شود را انتخاب کنید.")
        submit = st.form_submit_button("محاسبه ROIC")

    if submit:
        try:
            financials_period = handler.get_financials_for_period(period)
            balance_sheet = financials_period['balance_sheet']
            income_statement = financials_period['income_statement']

            if balance_sheet.empty or income_statement.empty:
                st.error(f"❌ داده‌های کافی برای دوره {period} وجود ندارد.")
                return

            # تشخیص بانک بودن شرکت
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

            # محاسبه کل بدهی
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
                st.error("⚠️ سرمایه‌گذاری صفر یا منفی است، امکان محاسبه ROIC وجود ندارد.")
                return

            operating_income = float(income_statement.get('Operating Profit (Loss)', 0) or 0.0)
            tax_rate_decimal = tax_rate / 100
            nopat = operating_income * (1 - tax_rate_decimal)
            roic = (nopat / invested_capital) * 100 if invested_capital != 0 else 0

            st.subheader(f"📊 ROIC: {format_percentage(roic)}")
            with st.expander("جزئیات محاسبه"):
                details = {
                    "کل بدهی": format_number(total_debt),
                    "کل حقوق صاحبان سهام": format_number(total_equity),
                    "وجه نقد": format_number(cash),
                    "سرمایه‌گذاری شده": format_number(invested_capital),
                    "سود عملیاتی (EBIT)": format_number(operating_income),
                    "NOPAT": format_number(nopat),
                    "ROIC (%)": format_percentage(roic)
                }
                st.table(pd.DataFrame(list(details.items()), columns=["پارامتر", "مقدار"]))

            # روند تاریخی ROIC
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
                fig = px.line(roic_df, x='Period', y='ROIC (%)', title='📈 روند تاریخی ROIC', markers=True)
                st.plotly_chart(fig, use_container_width=True)
                export_results_as_csv(roic_df, 'ROIC_Over_Periods.csv', "دانلود تاریخچه ROIC به CSV")
            else:
                st.info("⚠️ داده کافی برای رسم روند تاریخی ROIC وجود ندارد.")

        except Exception as e:
            st.error(f"❌ خطا در محاسبه ROIC: {e}")
            logging.error(f"ROIC Calculator Error: {e}")

    st.markdown("---")

def wacc_calculator(financials: Dict[str, pd.DataFrame]):
    show_wacc_page()
    st.markdown("""
    با این ابزار می‌توانید WACC را بر اساس داده‌های مالی شرکت محاسبه کنید.
    """)
    currency_options = ["IRR", "USD", "EUR", "GBP"]
    currency = st.selectbox("واحد پولی:", options=currency_options, index=0, key='wacc_currency',
                            help="انتخاب واحد پولی برای نمایش نتایج.")
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
        st.error("❌ داده‌های کافی وجود ندارد. لطفاً داده‌ها را بارگذاری کنید.")
        return

    outstanding_shares = st.number_input(
        "تعداد سهام منتشر شده:",
        min_value=0.0, value=3600000000.0, step=1000000.0,
        help="تعداد کل سهام شرکت را وارد کنید تا ارزش بازار محاسبه شود.",
        key="outstanding_shares"
    )

    with st.form("wacc_form"):
        st.subheader("📋 تنظیمات ورودی WACC")
        risk_free_rate_input = st.number_input("نرخ بدون ریسک (Rf) (%) :", min_value=0.0, value=3.0, step=0.1,
                                               help="نرخ اوراق قرضه دولتی بلندمدت به عنوان نرخ بدون ریسک.")
        market_return_input = st.number_input("بازده مورد انتظار بازار (Rm) (%) :", min_value=0.0, value=8.0, step=0.1,
                                              help="بازده مورد انتظار کل بازار.")
        tax_rate = st.number_input("نرخ مالیات شرکت (Tc) (%) :", min_value=0.0, value=21.0, step=0.1,
                                   help="نرخ مالیات موثر شرکت.")
        country_risk_premium = st.number_input("صرف ریسک کشور (%) :", min_value=0.0, value=0.2, step=0.01,
                                               help="صرف ریسک مربوط به کشور.")
        specific_risk_premium = st.number_input("صرف ریسک شرکت (%) :", min_value=0.0, value=0.35, step=0.01,
                                                help="صرف ریسک خاص شرکت.")
        size_premium = st.number_input("صرف ناشی از اندازه شرکت (%) :", min_value=0.0, value=0.3, step=0.01,
                                       help="صرف ریسک ناشی از کوچک بودن شرکت.")
        submit = st.form_submit_button("محاسبه WACC")

    if submit:
        try:
            financials_period = handler.get_financials_for_period(latest_period)
            balance_sheet = financials_period['balance_sheet']
            cashflow = financials_period['cashflow']
            keystats = financials_period['keystats']
            income_statement = financials_period['income_statement']

            if balance_sheet.empty or cashflow.empty or income_statement.empty:
                st.error(f"❌ داده‌های دوره {latest_period} ناقص است.")
                return

            # محاسبه بتا
            beta = keystats.get('Beta', None)
            if pd.isna(beta) or beta == 0:
                beta_calculated = handler.calculate_beta()
                if beta_calculated is not None:
                    beta = beta_calculated
                    st.info(f"✅ بتا از داده‌های تاریخی محاسبه شد: {beta:.4f}")
                else:
                    st.error("❌ عدم امکان تعیین بتا، لطفاً بتا را در داده‌های کلیدی وارد کنید.")
                    return
            else:
                beta = float(beta) if pd.notnull(beta) else None
                if beta is None:
                    st.error("❌ مقدار بتا در آمار کلیدی نامعتبر است.")
                    return

            # تشخیص بانک بودن شرکت
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

            # محاسبه کل بدهی
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
                st.error("⚠️ سرمایه‌گذاری صفر یا منفی است، محاسبه WACC ممکن نیست.")
                return

            interest_expense = float(income_statement.get('Interest Expense', 0) or 0.0)
            cost_of_debt = abs(interest_expense) / total_debt if total_debt != 0 else 0
            cost_of_debt_after_tax = cost_of_debt * (1 - tax_rate / 100)

            equity_risk_premium = (market_return_input - risk_free_rate_input) / 100
            cost_of_equity = ((risk_free_rate_input / 100) +
                               (beta * equity_risk_premium) +
                               (country_risk_premium / 100) +
                               (specific_risk_premium / 100) +
                               (size_premium / 100))

            df_prices = financials.get('historical_prices', pd.DataFrame())
            latest_price = df_prices.iloc[-1]['Close'] if not df_prices.empty else 0.0

            market_value_of_equity = (latest_price * st.session_state["outstanding_shares"]) / 1e6
            total_firm_value = market_value_of_equity + total_debt
            equity_weight = (market_value_of_equity / total_firm_value) if total_firm_value != 0 else 0
            debt_weight = (total_debt / total_firm_value) if total_firm_value != 0 else 0

            wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt_after_tax)

            if total_equity != 0:
                target_debt_equity_ratio = total_debt / total_equity
            else:
                target_debt_equity_ratio = 0.0

            if total_equity != 0:
                unlevered_beta = beta / (1 + ((1 - tax_rate / 100) * (total_debt / total_equity)))
            else:
                unlevered_beta = 0

            target_equity_weight = 1 / (1 + target_debt_equity_ratio)
            target_debt_weight = target_debt_equity_ratio / (1 + target_debt_equity_ratio)
            relevered_beta_target = unlevered_beta * (1 + ((1 - tax_rate / 100) * target_debt_equity_ratio))
            cost_of_equity_target = ((risk_free_rate_input / 100) +
                                     (relevered_beta_target * equity_risk_premium) +
                                     (country_risk_premium / 100) +
                                     (specific_risk_premium / 100) +
                                     (size_premium / 100))
            wacc_target = (target_equity_weight * cost_of_equity_target) + (target_debt_weight * cost_of_debt_after_tax)

            wacc_book = wacc

            st.subheader(f"📊 WACC: {format_percentage(wacc * 100)}")
            st.subheader(f"🎯 WACC هدف: {format_percentage(wacc_target * 100)}")
            st.subheader(f"📚 WACC دفتری: {format_percentage(wacc_book * 100)}")

            with st.expander("جزئیات محاسبه"):
                details = {
                    "نرخ بدون ریسک (Rf)": format_percentage(risk_free_rate_input),
                    "بازده بازار (Rm)": format_percentage(market_return_input),
                    "بتا": f"{beta:.4f}",
                    "صرف ریسک سهام (ERP)": format_percentage(equity_risk_premium * 100),
                    "صرف ریسک کشور": format_percentage(country_risk_premium),
                    "صرف ریسک شرکت": format_percentage(specific_risk_premium),
                    "صرف اندازه شرکت": format_percentage(size_premium),
                    "بتای بدون اهرم": f"{unlevered_beta:.4f}",
                    "بتای با اهرم هدف": f"{relevered_beta_target:.4f}",
                    "هزینه سهام (فعلی)": format_percentage(cost_of_equity * 100),
                    "هزینه سهام (هدف)": format_percentage(cost_of_equity_target * 100),
                    "هزینه بدهی پس از مالیات": format_percentage(cost_of_debt_after_tax * 100),
                    "وزن سهام (فعلی)": f"{equity_weight:.4f}",
                    "وزن بدهی (فعلی)": f"{debt_weight:.4f}",
                    "نسبت D/E فعلی": f"{total_debt/total_equity:.4f}" if total_equity != 0 else "N/A",
                    "نسبت D/E هدف": f"{target_debt_equity_ratio:.4f}",
                    "WACC (%)": format_percentage(wacc * 100),
                    "WACC هدف (%)": format_percentage(wacc_target * 100),
                    "WACC دفتری (%)": format_percentage(wacc_book * 100)
                }
                st.table(pd.DataFrame(list(details.items()), columns=["پارامتر", "مقدار"]))

            # تحلیل حساسیت WACC
            st.markdown("### 📊 تحلیل حساسیت")
            sensitivity_data = []
            cost_of_equity_variations = [cost_of_equity * 0.9, cost_of_equity, cost_of_equity * 1.1]
            cost_of_debt_variations = [cost_of_debt_after_tax * 0.9, cost_of_debt_after_tax, cost_of_debt_after_tax * 1.1]
            for coe in cost_of_equity_variations:
                for cod in cost_of_debt_variations:
                    wacc_calculated = (equity_weight * coe) + (debt_weight * cod)
                    sensitivity_data.append({
                        "هزینه سهام (%)": f"{coe * 100:.2f}",
                        "هزینه بدهی پس از مالیات (%)": f"{cod * 100:.2f}",
                        "WACC (%)": wacc_calculated * 100
                    })

            sensitivity_df = pd.DataFrame(sensitivity_data)
            if not sensitivity_df.empty:
                st.dataframe(sensitivity_df)
                try:
                    pivot_table = sensitivity_df.pivot(index="هزینه سهام (%)", columns="هزینه بدهی پس از مالیات (%)", values="WACC (%)")
                    fig = px.imshow(
                        pivot_table.astype(float),
                        labels=dict(x="هزینه بدهی پس از مالیات (%)", y="هزینه سهام (%)", color="WACC (%)"),
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        color_continuous_scale='Viridis',
                        title='📊 نمودار حساسیت WACC'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ خطا در رسم نمودار حساسیت: {e}")
                    logging.error(f"Sensitivity Analysis Heatmap Error: {e}")
            else:
                st.info("⚠️ داده کافی برای تحلیل حساسیت وجود ندارد.")

            # نمایش اجزای WACC
            wacc_components = pd.DataFrame({
                "جزء": ["سهام", "بدهی"],
                "وزن": [equity_weight, debt_weight],
                "هزینه (%)": [cost_of_equity * 100, cost_of_debt_after_tax * 100]
            })
            fig = px.bar(wacc_components, x='جزء', y='وزن', title='📊 اجزای WACC بر اساس وزن', text='وزن')
            st.plotly_chart(fig, use_container_width=True)

            # خروجی به CSV
            wacc_export_df = pd.DataFrame({
                "نوع WACC": ["WACC فعلی", "WACC هدف", "WACC دفتری"],
                "WACC (%)": [wacc * 100, wacc_target * 100, wacc_book * 100]
            })
            export_results_as_csv(wacc_export_df, 'WACC_Results.csv', "دانلود نتایج WACC به CSV")

        except Exception as e:
            st.error(f"❌ خطا در محاسبه WACC: {e}")
            logging.error(f"WACC Calculator Error: {e}")
    st.markdown("---")


def dcf_calculator(financials: Dict[str, Any]):
    show_dcf_page()
    st.markdown("""
    در این بخش می‌توانید ارزش ذاتی شرکت را با استفاده از روش DCF محاسبه کنید.
    """)
    handler = FinancialDataHandler(financials)
    latest_period = handler.get_latest_financial_period()
    if not latest_period:
        st.error("❌ داده‌های کافی وجود ندارد.")
        return

    financials_period = handler.get_financials_for_period(latest_period)
    cashflow = financials_period['cashflow']
    keystats = financials_period['keystats']

    if cashflow.empty:
        st.error(f"❌ داده‌های صورت جریان نقدینگی برای دوره {latest_period} موجود نیست.")
        return

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

    with st.form("dcf_wacc_form"):
        st.subheader("📋 تنظیمات (اختیاری)")
        user_wacc_input = st.number_input("WACC دلخواه (%) [0-100]:", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                          help="در صورت ورود مقداری بیشتر از 0، این مقدار جایگزین WACC محاسبه شده می‌شود.")
        submit_wacc = st.form_submit_button("اعمال WACC")

    if submit_wacc:
        if user_wacc_input > 0 and user_wacc_input <= 100:
            discount_rate = user_wacc_input / 100.0
            st.info(f"WACC وارد شده: {user_wacc_input:.2f}%")
        else:
            discount_rate = wacc_market
            st.warning("مقدار WACC وارد شده نامعتبر است. از WACC محاسبه شده استفاده می‌شود.")
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

    base_growth = historical_cagr
    base_discount = discount_rate
    base_tgr = terminal_growth_rate
    base_multiple = exit_multiple
    base_value, base_rows, base_tv, base_tv_pv = project_dcf(starting_fcf, base_growth, base_discount, base_tgr, base_multiple)

    optimistic_growth = base_growth * 1.1
    optimistic_discount = base_discount * 0.9
    optimistic_multiple = base_multiple * 1.1
    optimistic_value, _, _, _ = project_dcf(starting_fcf, optimistic_growth, optimistic_discount, base_tgr, optimistic_multiple)

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

    st.subheader("ارزش بنگاه (سناریوی پایه)")
    st.markdown(f"**ارزش بنگاه (DCF):** {format_number(base_value, currency_symbol)}")

    dcf_df = pd.DataFrame(base_rows)
    dcf_df['Projected FCF'] = dcf_df['Projected FCF'].apply(lambda x: format_number(x, currency_symbol))
    dcf_df['Present Value'] = dcf_df['Present Value'].apply(lambda x: format_number(x, currency_symbol))
    dcf_df['Discount Factor'] = dcf_df['Discount Factor'].apply(lambda x: f"{x:.4f}")
    st.table(dcf_df)

    st.markdown(f"**ارزش نهایی:** {format_number(base_tv, currency_symbol)}")
    st.markdown(f"**ارزش نهایی (ارزش فعلی):** {format_number(base_tv_pv, currency_symbol)}")
    st.markdown(f"**ارزش کل بنگاه:** {format_number(base_value, currency_symbol)}")

    # سناریوها
    st.markdown("### تحلیل سناریو")
    scenario_data = {
        "سناریو": ["بدبینانه", "پایه", "خوش‌بینانه"],
        "نرخ رشد (%)": [pessimistic_growth*100, base_growth*100, optimistic_growth*100],
        "نرخ تنزیل (%)": [pessimistic_discount*100, base_discount*100, optimistic_discount*100],
        "ضریب خروج (Exit Multiple)": [pessimistic_multiple, base_multiple, optimistic_multiple],
        f"ارزش بنگاه ({currency_symbol})": [pessimistic_value, base_value, optimistic_value]
    }
    scenario_df = pd.DataFrame(scenario_data)
    scenario_df["نرخ رشد (%)"] = scenario_df["نرخ رشد (%)"].apply(lambda x: format_percentage(x))
    scenario_df["نرخ تنزیل (%)"] = scenario_df["نرخ تنزیل (%)"].apply(lambda x: format_percentage(x))
    scenario_df["ضریب خروج (Exit Multiple)"] = scenario_df["ضریب خروج (Exit Multiple)"].apply(lambda x: f"{x:.2f}x")
    scenario_df[f"ارزش بنگاه ({currency_symbol})"] = scenario_df[f"ارزش بنگاه ({currency_symbol})"].apply(lambda x: format_number(x, currency_symbol))

    st.table(scenario_df)

    export_results_as_csv(pd.DataFrame(base_rows), 'DCF_Table.csv', "دانلود جدول DCF به CSV")
    export_results_as_csv(pd.DataFrame(scenario_data), 'Scenario_Analysis.csv', "دانلود تحلیل سناریو به CSV")

    st.markdown("---")
    st.markdown("**توجه:** مقادیر فوق از داده‌های تاریخی و WACC کاربر استخراج شده‌اند. مفروضات را در صورت نیاز تغییر دهید.")

def main():
    st.title("💡 محاسبه‌گرهای مالی")
    show_intro_page()

    st.sidebar.header("📂 بارگذاری داده‌های مالی")
    balance_sheet_file = st.sidebar.file_uploader("فایل اکسل ترازنامه", type=["xlsx", "xls"])
    cash_flow_file = st.sidebar.file_uploader("فایل اکسل صورت جریان نقدینگی", type=["xlsx", "xls"])
    income_statement_file = st.sidebar.file_uploader("فایل اکسل صورت سود و زیان", type=["xlsx", "xls"])
    keystats_file = st.sidebar.file_uploader("فایل اکسل آمار کلیدی", type=["xlsx", "xls"])
    historical_prices_file = st.sidebar.file_uploader("فایل اکسل قیمت‌های تاریخی سهام", type=["xlsx", "xls"])
    market_index_file = st.sidebar.file_uploader("فایل اکسل شاخص بازار", type=["xlsx", "xls"])
    risk_free_rate_file = st.sidebar.file_uploader("فایل اکسل نرخ بدون ریسک", type=["xlsx", "xls"])

    if st.sidebar.button("بارگذاری داده‌های مالی"):
        if not (balance_sheet_file and cash_flow_file and income_statement_file and keystats_file and historical_prices_file and market_index_file and risk_free_rate_file):
            st.sidebar.error("❌ لطفاً تمامی فایل‌های مورد نیاز را بارگذاری کنید.")
            st.stop()

        financial_data = load_excel_files(
            balance_sheet_file, cash_flow_file, income_statement_file,
            keystats_file, historical_prices_file, market_index_file, risk_free_rate_file
        )
        if financial_data:
            st.session_state['financial_data'] = financial_data
            st.sidebar.success("✅ داده‌های مالی با موفقیت بارگذاری شد.")
        else:
            st.sidebar.error("❌ بارگذاری داده‌های مالی ناموفق بود.")
            st.stop()

    if 'financial_data' not in st.session_state:
        st.info("🔍 لطفاً تمامی فایل‌ها را بارگذاری کرده و سپس دکمه 'بارگذاری داده‌های مالی' را بزنید.")
        st.stop()

    financials = st.session_state['financial_data']

    st.sidebar.header("🛠️ انتخاب محاسبه‌گر")
    calculator_tabs = ["ROIC", "WACC", "DCF"]
    selected_tab = st.sidebar.radio("یک محاسبه‌گر را انتخاب کنید:", options=calculator_tabs)

    st.markdown("---")

    with st.container():
        if selected_tab == "ROIC":
            roic_calculator(financials)
        elif selected_tab == "WACC":
            wacc_calculator(financials)
        elif selected_tab == "DCF":
            dcf_calculator(financials)
        else:
            st.error("❌ انتخاب نامعتبر.")

    st.markdown("""
    ---
    © 2024 ماشین حساب مالی. کلیه حقوق محفوظ است.
    """)


if __name__ == "__main__":
    main()
