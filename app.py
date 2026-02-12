import streamlit as st
import pandas as pd
import os
import glob
import plotly.express as px
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(page_title="스마트(SM) 수급 분석기", layout="wide")

# Paths (using relative path for Streamlit Cloud deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "KR_SM_Stock")
META_FILE_SECTOR = os.path.join(BASE_DIR, "업종분류.csv")
META_FILE_BASIC = os.path.join(BASE_DIR, "Basic stock info.csv")

# -----------------------------------------------------------------------------
# 2. Authentication
# -----------------------------------------------------------------------------
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_password():
    if st.session_state['password_input'] == "81052831":
        st.session_state['authenticated'] = True
        # Clear password from session state for security (optional but good practice)
        st.session_state['password_input'] = ""
    else:
        st.error("비밀번호가 올바르지 않습니다.")

def logout():
    st.session_state['authenticated'] = False
    st.rerun()

if not st.session_state['authenticated']:
    st.title("🔒 로그인")
    st.text_input("비밀번호를 입력하세요", type="password", key="password_input", on_change=check_password)
    st.button("로그인", on_click=check_password)
    st.stop() # Stop execution if not authenticated

# -----------------------------------------------------------------------------
# 3. Main App (Authenticated)
# -----------------------------------------------------------------------------

# Logout Button in Sidebar
st.sidebar.button("로그아웃", on_click=logout)

# -----------------------------------------------------------------------------
# 4. Data Loading & Processing
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60)
def load_and_process_data():
    """
    Loads daily SM files, Sector info, and Basic info.
    Returns:
        df_all: Combined daily dataframe
        sector_map: Dictionary of Code -> Sector Name
        basic_info: DataFrame with current price info
    """
    # 2.1 Load Daily Files
    all_files = glob.glob(os.path.join(DATA_DIR, "*_SM stock.csv"))
    if not all_files:
        return pd.DataFrame(), {}, pd.DataFrame()

    daily_dfs = []
    for f in all_files:
        try:
            # Parse Date from filename: YYYYMMDD_SM stock.csv
            fname = os.path.basename(f)
            date_str = fname.split("_")[0]
            
            # Read CSV (cp949)
            # Force string for code column (index 0)
            df = pd.read_csv(f, encoding='cp949', header=0, dtype={0: str})
            
            # Standardize Column Names by Index to avoid encoding issues
            # Col 0: Code, 1: Name, 2: SellVol, 3: BuyVol, 4: NetBuyVol, 
            # 5: SellAmt, 6: BuyAmt, 7: NetBuyAmt (Last column)
            if len(df.columns) >= 8:
                df.columns = [
                    'Code', 'Name', 
                    'SellVol', 'BuyVol', 'NetBuyVol', 
                    'SellAmt', 'BuyAmt', 'NetBuyAmt'
                ]
            else:
                # Fallback if structure varies, try to map safely
                # Assuming first is code, second is name, last is net buy amount
                new_cols = ['Code', 'Name'] + [f'Col_{i}' for i in range(2, len(df.columns)-1)] + ['NetBuyAmt']
                df.columns = new_cols
                
            df['Date'] = date_str
            daily_dfs.append(df)
        except Exception as e:
            # st.error(f"Error reading {f}: {e}") # Suppress individual file errors to avoid clutter
            continue

    if not daily_dfs:
        return pd.DataFrame(), {}, pd.DataFrame()

    df_all = pd.concat(daily_dfs, ignore_index=True)

    # Clean Numeric Columns (remove commas and convert to float)
    num_cols = ['SellVol', 'BuyVol', 'NetBuyVol', 'SellAmt', 'BuyAmt', 'NetBuyAmt']
    for col in num_cols:
        if col in df_all.columns:
            if df_all[col].dtype == object:
                df_all[col] = df_all[col].str.replace(',', '')
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0)
    
    # Unit Conversion: Won -> Billions (억)
    if 'NetBuyAmt' in df_all.columns:
        df_all['NetBuyAmt_100M'] = df_all['NetBuyAmt'] / 100000000
    else:
        df_all['NetBuyAmt_100M'] = 0

    if 'BuyAmt' in df_all.columns:
        df_all['BuyAmt_100M'] = df_all['BuyAmt'] / 100000000
    else:
        df_all['BuyAmt_100M'] = 0

    # 2.2 Load Metadata (Sector)
    sector_map = {}
    if os.path.exists(META_FILE_SECTOR):
        try:
            s_df = pd.read_csv(META_FILE_SECTOR, encoding='cp949', header=None, dtype={0: str})
            # Assume columns: 0=Code, 1=SectorName
            if len(s_df.columns) >= 2:
                sector_map = dict(zip(s_df.iloc[:, 0], s_df.iloc[:, 1]))
        except:
            pass

    # 2.3 Load Metadata (Basic Info)
    basic_info = pd.DataFrame()
    if os.path.exists(META_FILE_BASIC):
        try:
            # Read without assuming header names are correct due to encoding issues
            # Inspect: Col 0=Code, Col 1=Name, Col 4=Current Price
            basic_info = pd.read_csv(META_FILE_BASIC, encoding='cp949', header=0, dtype={0: str})
            
            # Rename columns by index to ensure safety
            if len(basic_info.columns) >= 5:
                # Create a clean map
                clean_basic = pd.DataFrame()
                clean_basic['Code'] = basic_info.iloc[:, 0]
                clean_basic['CurrentPrice'] = basic_info.iloc[:, 4] # Index 4 is Current Price
                
                # Clean 'CurrentPrice'
                if clean_basic['CurrentPrice'].dtype == object:
                    clean_basic['CurrentPrice'] = clean_basic['CurrentPrice'].str.replace(',', '')
                clean_basic['CurrentPrice'] = pd.to_numeric(clean_basic['CurrentPrice'], errors='coerce').fillna(0)
                
                basic_info = clean_basic
            else:
                basic_info = pd.DataFrame(columns=['Code', 'CurrentPrice'])

        except Exception as e:
            st.error(f"기본 정보 파일 로드 중 오류: {e}")
            basic_info = pd.DataFrame(columns=['Code', 'CurrentPrice'])

    # Map Sector to Main DF
    df_all['Sector'] = df_all['Code'].map(sector_map).fillna('Unknown')

    return df_all, sector_map, basic_info

# -----------------------------------------------------------------------------
# 5. Calculation Logic
# -----------------------------------------------------------------------------
def get_sorted_dates(df):
    return sorted(df['Date'].unique(), reverse=True)

def calc_sector_ranking(df, days):
    dates = get_sorted_dates(df)[:days]
    subset = df[df['Date'].isin(dates)]
    ranking = subset.groupby('Sector')['NetBuyAmt_100M'].sum().sort_values(ascending=False)
    # Convert Series to DataFrame for styling
    return ranking.to_frame(name='NetBuyAmt_100M')

def run_abc_strategy(df, basic_info):
    dates = get_sorted_dates(df)
    
    # Data Subsets
    df_20 = df[df['Date'].isin(dates[:20])]
    df_5 = df[df['Date'].isin(dates[:5])]
    df_3 = df[df['Date'].isin(dates[:3])]

    # A: Top 30 (20 days Sum)
    a_group = df_20.groupby(['Code', 'Name'])[['NetBuyAmt_100M', 'BuyAmt', 'BuyVol']].sum()
    top_30_a = a_group.sort_values('NetBuyAmt_100M', ascending=False).head(30)
    
    # Calculate Avg Price for A (Total Buy Amt / Total Buy Vol)
    # Note: Raw units (Won / Vol) -> Price in Won
    top_30_a['AvgPrice'] = top_30_a.apply(
        lambda x: x['BuyAmt'] / x['BuyVol'] if x['BuyVol'] > 0 else 0, axis=1
    )

    # B: Net Sell (5 days Sum < 0)
    b_group = df_5.groupby('Code')['NetBuyAmt_100M'].sum()
    b_exclude = b_group[b_group < 0].index.tolist()

    # C: Net Sell (3 days Sum < 0)
    c_group = df_3.groupby('Code')['NetBuyAmt_100M'].sum()
    c_exclude = c_group[c_group < 0].index.tolist()

    # Filter
    exclude_codes = set(b_exclude) | set(c_exclude)
    
    # Add status columns
    top_30_a['Status'] = top_30_a.index.get_level_values('Code').isin(exclude_codes)
    top_30_a['Reason'] = ''
    
    # Mark reasons for display
    def get_reason(code):
        reasons = []
        if code in b_exclude: reasons.append("5일 유출")
        if code in c_exclude: reasons.append("3일 유출")
        return ", ".join(reasons)
        
    top_30_a['Reason'] = [get_reason(c) for c in top_30_a.index.get_level_values('Code')]

    # Reset Index for merging
    res_df = top_30_a.reset_index()

    # Merge Current Price
    if not basic_info.empty:
        res_df = pd.merge(res_df, basic_info, on='Code', how='left')
        res_df['CurrentPrice'] = res_df['CurrentPrice'].fillna(0)
        
    # Calculate Disparity: (Current - Avg) / Avg * 100
    res_df['Disparity'] = res_df.apply(
        lambda x: ((x['CurrentPrice'] - x['AvgPrice']) / x['AvgPrice'] * 100) 
        if x['CurrentPrice'] > 0 and x['AvgPrice'] > 0 else 0.0, 
        axis=1
    )

    final = res_df[~res_df['Code'].isin(exclude_codes)].copy()
    excluded = res_df[res_df['Code'].isin(exclude_codes)].copy()
    
    return final, excluded

def get_consecutive_buys(df, days):
    dates = get_sorted_dates(df)[:days]
    subset = df[df['Date'].isin(dates)]
    
    # Pivot: Index=Code, Col=Date, Val=NetBuy
    pivot = subset.pivot_table(index=['Code', 'Name'], columns='Date', values='NetBuyAmt_100M')
    
    # Check if all columns > 0
    # Also handle missing data (NaN) as not buying -> strict consecutive check means no NaNs and > 0
    cond = (pivot > 0).all(axis=1) & (pivot.notna().all(axis=1))
    
    res = pivot[cond].copy()
    res['Total_NetBuy'] = res.sum(axis=1)
    
    return res.sort_values('Total_NetBuy', ascending=False).head(10)

# -----------------------------------------------------------------------------
# 6. UI Layout (Main Dashboard)
# -----------------------------------------------------------------------------
st.sidebar.title("🚀 분석 제어판")
if st.sidebar.button("분석 실행 (데이터 갱신)"):
    st.cache_data.clear()
    st.rerun()

# Load Data
df, sector_map, basic_info = load_and_process_data()

if df.empty:
    st.warning("KR_SM_Stock 폴더에 데이터가 없습니다. 경로와 파일 형식을 확인해주세요.")
    st.stop()

st.title("📈 스마트(SM) 전략 대시보드")
tabs = st.tabs(["🏭 업종 분석", "🎯 A-B-C 전략", "🔥 연속 순매수", "🔎 개별 종목 분석"])

# --- Tab 1: Sector ---
with tabs[0]:
    st.header("업종별 스마트 수급 현황 (단위: 억)")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("최근 1일 상위")
        st.dataframe(calc_sector_ranking(df, 1).head(10).style.format("{:,.2f} 억"))
    with c2:
        st.subheader("최근 3일 상위")
        st.dataframe(calc_sector_ranking(df, 3).head(10).style.format("{:,.2f} 억"))
    with c3:
        st.subheader("최근 5일 상위")
        s5 = calc_sector_ranking(df, 5)
        st.dataframe(s5.head(10).style.format("{:,.2f} 억"))
        
    st.divider()
    st.subheader("🌟 추천 유망 섹터 (5일 누적 상위 2개)")
    if len(s5) >= 2:
        top_sectors = s5.index[:2].tolist()
        st.success(f"1. {top_sectors[0]}   |   2. {top_sectors[1]}")

    st.markdown("### 📊 상위 5개 업종 차트 (5일 누적)")
    if not s5.empty:
        fig_s5 = px.bar(s5.head(5).reset_index(), x='Sector', y='NetBuyAmt_100M', title="상위 5개 업종 순매수 (5일)", color='NetBuyAmt_100M', labels={'Sector': '업종명', 'NetBuyAmt_100M': '순매수대금(억)'})
        st.plotly_chart(fig_s5, use_container_width=True)

# --- Tab 2: A-B-C ---
with tabs[1]:
    st.header("A-B-C 필터링 전략")
    st.info("전략 설명: A그룹(최근 20일 스마트 순매수 상위 30위) 중에서 B(최근 5일 유출)와 C(최근 3일 유출) 종목을 제외합니다.")
    
    final, excluded = run_abc_strategy(df, basic_info)
    
    # Display Config
    disp_cols = ['Code', 'Name', 'NetBuyAmt_100M', 'AvgPrice', 'CurrentPrice', 'Disparity']
    
    # Rename columns for display
    final_disp = final[disp_cols].rename(columns={
        'Code': '종목코드',
        'Name': '종목명',
        'NetBuyAmt_100M': '순매수대금(억)',
        'AvgPrice': '스마트평단가',
        'CurrentPrice': '현재가',
        'Disparity': '괴리율(%)'
    })
    
    excluded_disp = excluded[['Code', 'Name', 'Reason', 'NetBuyAmt_100M']].rename(columns={
        'Code': '종목코드',
        'Name': '종목명',
        'Reason': '제외사유',
        'NetBuyAmt_100M': '순매수대금(억)'
    })

    fmt = {
        '순매수대금(억)': '{:,.2f} 억',
        '스마트평단가': '{:,.0f} 원',
        '현재가': '{:,.0f} 원',
        '괴리율(%)': '{:,.2f} %'
    }

    # Ensure final is a DataFrame
    if isinstance(final_disp, pd.Series):
        final_disp = final_disp.to_frame()
    
    # Ensure excluded is a DataFrame
    if isinstance(excluded_disp, pd.Series):
        excluded_disp = excluded_disp.to_frame()

    st.subheader(f"✅ 최종 선정 종목 ({len(final)}개)")
    st.dataframe(final_disp.style.format(fmt, subset=['순매수대금(억)', '스마트평단가', '현재가', '괴리율(%)']))
    
    st.subheader(f"❌ 제외된 종목 (최근 스마트 자금 유출, {len(excluded)}개)")
    st.dataframe(excluded_disp.style.format({'순매수대금(억)': '{:,.2f} 억'}))

# --- Tab 3: Consecutive ---
with tabs[2]:
    st.header("연속 순매수 종목 (스마트 자금 지속 유입)")
    periods = [1, 3, 5, 7]
    
    cols = st.columns(len(periods))
    for i, p in enumerate(periods):
        with cols[i]:
            st.markdown(f"**최근 {p}일 연속**")
            res = get_consecutive_buys(df, p)
            if not res.empty:
                # Show Code/Name and Total Sum
                show = res[['Total_NetBuy']].reset_index().rename(columns={
                    'Code': '종목코드', 'Name': '종목명', 'Total_NetBuy': '기간합계(억)'
                })
                st.dataframe(show.style.format({'기간합계(억)': '{:,.2f} 억'}))
            else:
                st.write("- 해당 없음 -")

# --- Tab 4: Individual ---
with tabs[3]:
    st.header("개별 종목 상세 분석")
    
    # Search Box
    all_stocks = df[['Code', 'Name']].drop_duplicates()
    all_stocks['Label'] = all_stocks['Name'] + " (" + all_stocks['Code'] + ")"
    selection = st.selectbox("종목 검색", all_stocks['Label'].unique())
    
    if selection:
        code = selection.split("(")[-1].strip(")")
        
        # Filter Data (Last 7 Days)
        dates = get_sorted_dates(df)[:7]
        target = df[(df['Code'] == code) & (df['Date'].isin(dates))].sort_values('Date')
        
        if not target.empty:
            st.subheader(f"{selection}")
            
            # Metrics
            total_buy = target['NetBuyAmt_100M'].sum()
            col1, col2 = st.columns(2)
            col1.metric("최근 7일 스마트 순매수 합계", f"{total_buy:,.2f} 억")
            
            # Chart
            fig = px.bar(
                target, x='Date', y='NetBuyAmt_100M',
                title="일별 스마트 순매수 추이 (단위: 억)",
                text_auto='.2f',
                color='NetBuyAmt_100M',
                color_continuous_scale='Bluered',
                labels={'Date': '날짜', 'NetBuyAmt_100M': '순매수대금(억)'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            target_disp = target[['Date', 'NetBuyAmt_100M', 'BuyVol', 'SellVol', 'NetBuyAmt']].rename(columns={
                'Date': '날짜',
                'NetBuyAmt_100M': '순매수대금(억)',
                'BuyVol': '매수량',
                'SellVol': '매도량',
                'NetBuyAmt': '순매수대금(원)'
            })
            
            st.dataframe(
                target_disp.style.format({
                    '순매수대금(억)': '{:,.2f} 억',
                    '매수량': '{:,.0f}',
                    '매도량': '{:,.0f}', 
                    '순매수대금(원)': '{:,.0f}'
                })
            )
        else:
            st.info("최근 7일간의 데이터가 없습니다.")
