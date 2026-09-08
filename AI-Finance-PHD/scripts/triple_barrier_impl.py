import numpy as np
import pandas as pd

def get_daily_vol(close, span0=100):
    """
    Marcos Lopez de Prado - Advances in Financial Machine Learning, Chapter 3, page 44.
    Computes the daily volatility of close prices using an exponentially weighted moving standard deviation.
    """
    # 1. Compute daily returns
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    
    # 2. Compute exponentially weighted moving standard deviation
    df0 = df0.ewm(span=span0).std()
    return df0

def get_vertical_barriers(close, t_events, num_days=1):
    """
    Advances in Financial Machine Learning, Chapter 3, page 44.
    Finds the timestamp of the vertical barrier (expiration) for each event.
    """
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])  # NaNs at the end are handled
    return t1

def apply_pt_sl_on_t1(close, events, pt_sl, target):
    """
    Advances in Financial Machine Learning, Chapter 3, page 45.
    Finds the timestamp of the first barrier touched (profit taking, stop loss, or vertical barrier).
    
    Parameters:
    - close: pandas Series of close prices
    - events: pandas DataFrame with columns:
              - 't1': vertical barrier timestamp (can be NaNs)
              - 'trgt': dynamic target (volatility-based threshold)
              - 'side': (optional) 1 for buy, -1 for sell. If not present, assumed to be 1.
    - pt_sl: non-negative float array [pt, sl] indicating multipliers for target to set barriers.
             pt_sl[0] for profit taking, pt_sl[1] for stop loss.
    - target: pandas Series of targets (typically daily volatility)
    """
    # Apply profit taking and stop loss thresholds
    out = events[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * target
    else:
        pt = pd.Series(index=events.index, dtype=float)  # No profit taking barrier
        
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * target
    else:
        sl = pd.Series(index=events.index, dtype=float)  # No stop loss barrier
        
    # Check if 'side' exists in events; if not, default to 1 (long only)
    if 'side' in events:
        side = events['side']
    else:
        side = pd.Series(1., index=events.index)
        
    for loc, t_ind in events['t1'].fillna(close.index[-1]).items():
        path = close.loc[loc:t_ind]  # path of prices during the holding period
        returns = (path / close.loc[loc] - 1) * side.loc[loc]  # path returns adjusted for side
        
        # Find first touch of profit taking or stop loss
        pt_barrier = pt.loc[loc] if not pd.isna(pt.loc[loc]) else np.inf
        sl_barrier = sl.loc[loc] if not pd.isna(sl.loc[loc]) else -np.inf
        
        first_pt = returns[returns >= pt_barrier].index.min()
        first_sl = returns[returns <= sl_barrier].index.min()
        
        # Store the earliest touch timestamp
        if first_pt is not None or first_sl is not None:
            if first_pt is None:
                out.loc[loc, 't1'] = first_sl
            elif first_sl is None:
                out.loc[loc, 't1'] = first_pt
            else:
                out.loc[loc, 't1'] = min(first_pt, first_sl)
    return out

def get_bins(events, close):
    """
    Advances in Financial Machine Learning, Chapter 3, page 46.
    Generates labels (bins) based on which barrier was touched first.
    
    Returns DataFrame with:
    - 'ret': return at the first touched barrier
    - 'bin': label (-1 for stop loss, 1 for profit taking, 0 for vertical barrier/expiration)
             If 'side' is present, labels can be binary {0, 1} for Meta-Labeling!
    """
    # 1. Retrieve prices at exit points
    events_ = events.dropna(subset=['t1'])
    px = events_['t1'].values
    px = close.loc[px].values
    
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px / close.loc[events_.index].values - 1
    
    if 'side' in events_:
        out['ret'] *= events_['side']  # adjust return for short positions
        
    out['bin'] = np.sign(out['ret'])
    
    # If meta-labeling: bin is 1 if prediction was successful (profit taken), 0 otherwise (stop loss or vertical barrier)
    if 'side' in events_:
        # For meta-labeling:
        # If return is positive (profit taken), bin = 1
        # If return is negative/zero (stop loss or time-out), bin = 0
        out['bin'] = (out['ret'] > 0).astype(int)
        
    return out

# ─── DEMONSTRATION & TESTING BLOCK ───
if __name__ == "__main__":
    print("Running Triple-Barrier Method Simulation...")
    # Generate synthetic price data
    np.random.seed(42)
    idx = pd.date_range(start="2026-01-01", periods=1000, freq="h")
    returns = np.random.normal(loc=0.0001, scale=0.01, size=1000)
    prices = 100 * np.exp(np.cumsum(returns))
    close = pd.Series(prices, index=idx)
    
    # 1. Compute dynamic target (daily volatility)
    print("1. Computing daily volatility as dynamic target...")
    target = get_daily_vol(close, span0=100)
    
    # Drop NaNs from target
    target = target.dropna()
    t_events = target.index  # Here we treat all indices with valid targets as events for demonstration
    
    # 2. Compute vertical barriers (max holding period: 2 days)
    print("2. Generating vertical barriers (2 days)...")
    t1 = get_vertical_barriers(close, t_events, num_days=2)
    
    # 3. Form events DataFrame
    events = pd.DataFrame(index=t_events)
    events['t1'] = t1
    events['trgt'] = target
    # Let's add a random 'side' (-1 or 1) to simulate buy/sell signals for Meta-Labeling
    events['side'] = np.random.choice([-1, 1], size=len(events))
    
    # 4. Apply Triple Barrier (pt_sl=[1.0, 1.0] -> 1x Volatility for PT and SL)
    print("4. Applying Triple Barrier Method...")
    pt_sl = [1.0, 1.0]
    barrier_touches = apply_pt_sl_on_t1(close, events, pt_sl, target)
    
    # Merge exit timestamps back to events
    events['t1'] = barrier_touches['t1']
    
    # 5. Generate Labels (Meta-Labeling style: {0, 1})
    print("5. Generating Meta-Labels...")
    labels = get_bins(events, close)
    
    print("\n--- SAMPLE OUTPUT DATA ---")
    print(labels.head(10))
    print("\nLabel Distribution (Meta-Labeling):")
    print(labels['bin'].value_counts())

