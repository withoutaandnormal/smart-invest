from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import os
import glob
import json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'smart_money_secret_key_change_this' # Session key
PASSWORD = "1234" # Default Password (Change if needed)

# Login Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Paths (Updated for flat structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_SM = os.path.join(BASE_DIR, "KR_SM_Stock")
DATA_DIR_STOCK = os.path.join(BASE_DIR, "KR_Stock")
META_FILE_SECTOR = os.path.join(BASE_DIR, "업종분류.csv")
META_FILE_BASIC = os.path.join(BASE_DIR, "Basic stock info.csv")

# Global Data Cache
CACHED_DATA = {}

def load_data():
    """Loads daily SM files, Sector info, and Basic info into global CACHED_DATA."""
    global CACHED_DATA
    print("Loading data...")
    
    # 1. Load Daily SM Files
    all_files = glob.glob(os.path.join(DATA_DIR_SM, "*_SM stock.csv"))
    if not all_files:
        print(f"Warning: No SM files found in {DATA_DIR_SM}")
        CACHED_DATA = {} # Reset or keep empty
        return

    daily_dfs = []
    for f in all_files:
        try:
            fname = os.path.basename(f)
            date_str = fname.split("_")[0]
            df = pd.read_csv(f, encoding='cp949', header=0, dtype={0: str})
            
            # Standardize Columns
            if len(df.columns) >= 8:
                df.columns = ['Code', 'Name', 'SellVol', 'BuyVol', 'NetBuyVol', 'SellAmt', 'BuyAmt', 'NetBuyAmt']
            else:
                new_cols = ['Code', 'Name'] + [f'Col_{i}' for i in range(2, len(df.columns)-1)] + ['NetBuyAmt']
                df.columns = new_cols
            
            df['Date'] = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
            daily_dfs.append(df)
        except:
            continue

    if not daily_dfs:
        df_all = pd.DataFrame()
    else:
        df_all = pd.concat(daily_dfs, ignore_index=True)

    # Clean Numeric
    num_cols = ['SellVol', 'BuyVol', 'NetBuyVol', 'SellAmt', 'BuyAmt', 'NetBuyAmt']
    for col in num_cols:
        if col in df_all.columns:
            if df_all[col].dtype == object:
                df_all[col] = df_all[col].str.replace(',', '')
            df_all[col] = pd.to_numeric(df_all[col], errors='coerce').fillna(0)
    
    # Unit Conversion
    if 'NetBuyAmt' in df_all.columns:
        df_all['NetBuyAmt_100M'] = df_all['NetBuyAmt'] / 100000000
    if 'BuyAmt' in df_all.columns:
        df_all['BuyAmt_100M'] = df_all['BuyAmt'] / 100000000

    # 2. Sector Metadata
    sector_map = {}
    if os.path.exists(META_FILE_SECTOR):
        try:
            s_df = pd.read_csv(META_FILE_SECTOR, encoding='cp949', header=0, dtype=str)
            s_df.iloc[:, 0] = s_df.iloc[:, 0].str.replace('"', '').str.strip()
            s_df.iloc[:, 3] = s_df.iloc[:, 3].str.replace('"', '').str.strip()
            valid_sectors = s_df[s_df.iloc[:, 3].notna() & (s_df.iloc[:, 3] != '')]
            sector_map = dict(zip(valid_sectors.iloc[:, 0], valid_sectors.iloc[:, 3]))
        except Exception as e:
            print(f"Error loading sector map: {e}")
            pass

    # 3. Basic Info
    basic_info = pd.DataFrame()
    if os.path.exists(META_FILE_BASIC):
        try:
            basic_info = pd.read_csv(META_FILE_BASIC, encoding='cp949', header=0, dtype={0: str})
            if len(basic_info.columns) >= 5:
                clean_basic = pd.DataFrame()
                clean_basic['Code'] = basic_info.iloc[:, 0]
                clean_basic['Name'] = basic_info.iloc[:, 1]
                clean_basic['CurrentPrice'] = basic_info.iloc[:, 4]
                if clean_basic['CurrentPrice'].dtype == object:
                    clean_basic['CurrentPrice'] = clean_basic['CurrentPrice'].str.replace(',', '')
                clean_basic['CurrentPrice'] = pd.to_numeric(clean_basic['CurrentPrice'], errors='coerce').fillna(0)
                basic_info = clean_basic
        except:
            pass

    df_all['Sector'] = df_all['Code'].map(sector_map).fillna('Unknown')
    
    # 4. Preload Prices (Optimization)
    price_cache = {}
    stock_files = glob.glob(os.path.join(DATA_DIR_STOCK, "*_All stock information.csv"))
    
    for f in stock_files:
        try:
            fname = os.path.basename(f)
            date_str = fname.split("_")[0]
            pdf = pd.read_csv(f, encoding='cp949', header=0, dtype=str, usecols=[0, 4])
            if len(pdf.columns) < 2: continue
            
            pdf.iloc[:, 0] = pdf.iloc[:, 0].str.replace('"', '').str.strip()
            pdf.iloc[:, 1] = pdf.iloc[:, 1].str.replace('"', '').str.replace(',', '').str.strip()
            
            daily_prices = dict(zip(pdf.iloc[:, 0], pdf.iloc[:, 1]))
            
            for code, price_str in daily_prices.items():
                try:
                    price = float(price_str)
                    if code not in price_cache:
                        price_cache[code] = {}
                    price_cache[code][date_str] = price
                except:
                    continue
        except:
            continue
            
    print("Data load complete.")
    
    CACHED_DATA = {
        'df': df_all,
        'sector_map': sector_map,
        'basic_info': basic_info,
        'sorted_dates': sorted(df_all['Date'].dropna().unique(), reverse=True) if not df_all.empty else [],
        'prices': price_cache
    }

# Load data on start
load_data()

# Refresh Data Route
@app.route('/refresh')
@login_required
def refresh_data():
    load_data()
    return redirect(url_for('index'))

# Helper Functions
def get_stock_price_history(code):
    # Optimized: Read from memory cache
    price_cache = CACHED_DATA.get('prices', {})
    code = code.replace('"', '').strip()
    
    if code not in price_cache:
        return []
        
    history = []
    for date_str, price in price_cache[code].items():
        history.append({'Date': date_str, 'Close': price})
        
    return sorted(history, key=lambda x: x['Date'])

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['password'] == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = '비밀번호가 틀렸습니다.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', page='home')

@app.route('/sector')
@login_required
def sector():
    # Use CACHED_DATA directly for speed
    df = CACHED_DATA.get('df', pd.DataFrame())
    dates = CACHED_DATA.get('sorted_dates', [])
    days = int(request.args.get('days', 5))
    
    if df.empty or not dates:
        return render_template('index.html', page='sector', data=[], days=days, top_picks={})

    # 1. Target Dates
    target_dates = dates[:days]
    subset = df[df['Date'].isin(target_dates)].copy()
    
    # 2. Add Disparity Info (Need Price)
    # Group by Stock first to get Period Sum
    stock_grp = subset.groupby(['Code', 'Name', 'Sector'])['NetBuyAmt'].sum().reset_index()
    stock_grp['IsPlus'] = stock_grp['NetBuyAmt'] > 0
    
    # 3. Sector Aggregation
    sector_stats = stock_grp.groupby('Sector').agg(
        Total_NetBuy=('NetBuyAmt', 'sum'),
        Count=('Code', 'count'),
        Plus_Count=('IsPlus', 'sum')
    ).reset_index()
    
    # 4. Calculate Indicators
    sector_stats['NetBuyAmt_100M'] = sector_stats['Total_NetBuy'] / 100000000
    sector_stats['Breadth'] = (sector_stats['Plus_Count'] / sector_stats['Count']) * 100
    
    # 5. Ranking (Sort by Amount desc, Exclude Unknown)
    ranking = sector_stats[sector_stats['Sector'] != 'Unknown'].sort_values('Total_NetBuy', ascending=False).head(10)
    ranking['NetBuyAmt_100M'] = ranking['NetBuyAmt_100M'].apply(lambda x: round(x, 2))
    ranking['Breadth'] = ranking['Breadth'].apply(lambda x: round(x, 1))
    
    # 6. Top Picks per Sector (Top 5 stocks by amount)
    top_picks = {}
    for sec in ranking['Sector']:
        # Get stocks in this sector from period sum data
        sec_stocks = stock_grp[stock_grp['Sector'] == sec].sort_values('NetBuyAmt', ascending=False).head(5)
        
        # Calculate Disparity for these top picks (Heavy lifting here only)
        picks_list = []
        for _, row in sec_stocks.iterrows():
            code = row['Code']
            
            # Get Price & Smart Avg (20 days from original logic)
            sm_20 = df[(df['Code'] == code) & (df['Date'].isin(dates[:20]))]
            
            s_sum = sm_20['BuyAmt'].sum()
            v_sum = sm_20['BuyVol'].sum()
            smart_avg = s_sum / v_sum if v_sum > 0 else 0
            
            # Get Current Price from cache
            price_hist = get_stock_price_history(code) 
            curr_price = price_hist[-1]['Close'] if price_hist else 0
            
            disp = 0
            if curr_price > 0 and smart_avg > 0:
                disp = (curr_price - smart_avg) / smart_avg * 100
                
            picks_list.append({
                'code': code,
                'name': row['Name'],
                'sector': sec,  # Added Sector Name
                'net_buy': round(row['NetBuyAmt'] / 100000000, 2),
                'price': curr_price,
                'disparity': round(disp, 2)
            })
        top_picks[sec] = picks_list

    data = ranking.to_dict(orient='records')
    return render_template('index.html', page='sector', data=data, days=days, top_picks=top_picks)

@app.route('/consecutive')
@login_required
def consecutive():
    # Use CACHED_DATA directly
    df = CACHED_DATA.get('df', pd.DataFrame())
    dates = CACHED_DATA.get('sorted_dates', [])
    days = int(request.args.get('days', 3))
    
    if df.empty or not dates:
        return render_template('index.html', page='consecutive', data=[], days=days)

    target_dates = dates[:days]
    subset = df[df['Date'].isin(target_dates)]
    
    # Pivot: Index=[Code, Name, Sector], Columns=Date, Values=NetBuyAmt
    pivot = subset.pivot_table(index=['Code', 'Name', 'Sector'], columns='Date', values='NetBuyAmt')
    
    # Condition: All values > 0 and Not Null
    cond = (pivot > 0).all(axis=1) & (pivot.notna().all(axis=1))
    res = pivot[cond].copy()
    
    # Calculate Stats
    res['Total_NetBuy'] = res.sum(axis=1)
    res = res.reset_index()
    
    # Add Disparity
    final_list = []
    for _, row in res.iterrows():
        code = row['Code']
        
        # Calculate Period Avg Price
        stock_data = subset[subset['Code'] == code]
        total_buy_amt = stock_data['BuyAmt'].sum()
        total_buy_vol = stock_data['BuyVol'].sum()
        
        smart_avg = total_buy_amt / total_buy_vol if total_buy_vol > 0 else 0
        
        # Get Current Price
        price_hist = get_stock_price_history(code)
        curr_price = price_hist[-1]['Close'] if price_hist else 0
        
        disp = 0
        if curr_price > 0 and smart_avg > 0:
            disp = (curr_price - smart_avg) / smart_avg * 100
            
        final_list.append({
            'code': code,
            'name': row['Name'],
            'sector': row['Sector'],
            'total_net': round(row['Total_NetBuy'] / 100000000, 2),
            'smart_avg': round(smart_avg),
            'curr_price': curr_price,
            'disparity': round(disp, 2)
        })
        
    # Sort by Total Net Buy Amount Descending
    final_list.sort(key=lambda x: x['total_net'], reverse=True)
        
    return render_template('index.html', page='consecutive', data=final_list, days=days)

@app.route('/abc')
@login_required
def abc():
    # Use CACHED_DATA directly
    df = CACHED_DATA.get('df', pd.DataFrame())
    dates = CACHED_DATA.get('sorted_dates', [])
    
    if df.empty or not dates:
        return render_template('index.html', page='abc', final=[], excluded=[])

    # 1. A Group (20 days): Top 30 Net Buy Amount
    dates_20 = dates[:20]
    df_20 = df[df['Date'].isin(dates_20)]
    a_ranking = df_20.groupby(['Code', 'Name'])['NetBuyAmt'].sum().sort_values(ascending=False).head(30)
    a_codes = a_ranking.index.get_level_values('Code').tolist() # Candidates

    # 2. B Group (5 days): Top 30 Net Sell Amount (Largest negative values)
    dates_5 = dates[:5]
    df_5 = df[df['Date'].isin(dates_5)]
    # Filter negative only, then sort by amount (ascending because negative is smaller)
    b_ranking = df_5.groupby('Code')['NetBuyAmt'].sum()
    b_ranking = b_ranking[b_ranking < 0].sort_values(ascending=True).head(30)
    b_codes = b_ranking.index.tolist()

    # 3. C Group (3 days): All Net Sell (< 0)
    dates_3 = dates[:3]
    df_3 = df[df['Date'].isin(dates_3)]
    c_ranking = df_3.groupby('Code')['NetBuyAmt'].sum()
    c_codes = c_ranking[c_ranking < 0].index.tolist()

    # 4. Filtering
    final_list = []
    excluded_list = []
    
    for (code, name), net_buy in a_ranking.items():
        reason = []
        if code in b_codes:
            reason.append("B: 5일 대량유출")
        if code in c_codes:
            reason.append("C: 3일 순매도")
            
        # Common Data (Price, Disparity)
        # Using A period (20 days) for Avg Price
        s_sum = df_20[df_20['Code'] == code]['BuyAmt'].sum()
        v_sum = df_20[df_20['Code'] == code]['BuyVol'].sum()
        smart_avg = s_sum / v_sum if v_sum > 0 else 0
        
        price_hist = get_stock_price_history(code)
        curr_price = price_hist[-1]['Close'] if price_hist else 0
        
        disp = 0
        if curr_price > 0 and smart_avg > 0:
            disp = (curr_price - smart_avg) / smart_avg * 100
            
        item = {
            'code': code,
            'name': name,
            'net_buy': round(net_buy / 100000000, 2),
            'smart_avg': round(smart_avg),
            'price': curr_price,
            'disparity': round(disp, 2),
            'reason': ", ".join(reason)
        }
        
        if reason:
            excluded_list.append(item)
        else:
            final_list.append(item)
            
    # Sort Final by 20-day Net Buy Amount Descending (Amount is King)
    final_list.sort(key=lambda x: x['net_buy'], reverse=True)
            
    return render_template('index.html', page='abc', final=final_list, excluded=excluded_list)

@app.route('/stock_api')
@login_required
def stock_api():
    code = request.args.get('code')
    if not code: return jsonify({})
    
    # Use CACHED_DATA directly
    df = CACHED_DATA.get('df', pd.DataFrame())
    dates = CACHED_DATA.get('sorted_dates', [])
    
    if df.empty or not dates: return jsonify({})

    # 1. Get last 20 business days (files)
    target_dates = dates[:20]
    target_dates.sort() # Sort ascending for chart
    
    # 2. Filter data
    sm_data = df[(df['Code'] == code) & (df['Date'].isin(target_dates))].sort_values('Date')
    
    # 3. Calculate Disparity (20 days sum)
    total_net_buy = sm_data['NetBuyAmt'].sum()
    total_buy_amt = sm_data['BuyAmt'].sum()
    total_buy_vol = sm_data['BuyVol'].sum()
    
    smart_avg_price = 0
    if total_buy_vol > 0:
        smart_avg_price = total_buy_amt / total_buy_vol
        
    # 4. Get Price History
    price_hist = get_stock_price_history(code)
    price_map = {x['Date']: x['Close'] for x in price_hist}
    
    # Match prices to dates
    chart_dates = [d.strftime('%Y-%m-%d') for d in target_dates]
    chart_prices = []
    current_price = 0
    
    for d_str in chart_dates:
        # Try exact match (YYYY-MM-DD -> YYYYMMDD)
        key = d_str.replace('-', '')
        val = price_map.get(key, 0)
        chart_prices.append(val)
        if val > 0: current_price = val
        
    disparity = 0
    if current_price > 0 and smart_avg_price > 0:
        disparity = (current_price - smart_avg_price) / smart_avg_price * 100

    return jsonify({
        'dates': chart_dates,
        'net_buy': sm_data['NetBuyAmt_100M'].tolist(),
        'prices': chart_prices,
        'info': {
            'current_price': current_price,
            'smart_avg': round(smart_avg_price),
            'disparity': round(disparity, 2),
            'total_net_buy': round(total_net_buy / 100000000, 2)
        }
    })

@app.route('/metadata')
def metadata():
    dates = CACHED_DATA.get('sorted_dates', [])
    last_date = dates[0].strftime('%Y-%m-%d') if dates else "-"
    return jsonify({'last_updated': last_date})

@app.route('/search_stock')
def search_stock():
    query = request.args.get('q', '').strip()
    if not query: return jsonify([])
    
    df = CACHED_DATA.get('df', pd.DataFrame())
    if df.empty: return jsonify([])
    
    # Get unique stocks
    stocks = df[['Code', 'Name']].drop_duplicates()
    
    # Search by Name or Code
    mask = stocks['Name'].str.contains(query) | stocks['Code'].str.contains(query)
    results = stocks[mask].head(10).to_dict(orient='records')
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
