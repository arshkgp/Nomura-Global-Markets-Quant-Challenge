# NOMURA QUANT CHALLENGE 2025
# Complete implementations for Task1, Task2 and Task3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# BACKTESTERS
# -----------------------------------------------------------------------------

def backtester_without_TC(weights_df):
    data = pd.read_csv('crossval_data.csv')
    weights_df = weights_df.fillna(0)

    # build returns matrix
    df_returns = pd.DataFrame({
        sym: data.loc[data.Symbol==sym, 'Close']
                    .reset_index(drop=True)
                    .pct_change()
                    .fillna(0)
        for sym in range(20)
    })
    # align dates
    dates = data.loc[data.Symbol==0, 'Date'].reset_index(drop=True)
    df_returns['Date'] = dates
    df_returns.set_index('Date', inplace=True)

    # slice
    start_date, end_date = 3500, 3999
    df_ret = df_returns.loc[start_date:end_date]
    w = weights_df.reindex(df_ret.index, fill_value=0)

    # daily portfolio returns
    pnl = (w * df_ret).sum(axis=1)
    notional = 1.0
    net_returns = []
    for r in pnl:
        net_returns.append(r)
        notional *= (1 + r)

    net_return_pct = (notional - 1.0) * 100
    sharpe = np.mean(net_returns) / np.std(net_returns)
    return [net_return_pct, sharpe]


def backtester_with_TC(weights_df, cost_rate=0.01):
    data = pd.read_csv('crossval_data.csv')
    weights_df = weights_df.fillna(0)

    # build returns matrix
    df_returns = pd.DataFrame({
        sym: data.loc[data.Symbol==sym, 'Close']
                    .reset_index(drop=True)
                    .pct_change()
                    .fillna(0)
        for sym in range(20)
    })
    dates = data.loc[data.Symbol==0, 'Date'].reset_index(drop=True)
    df_returns['Date'] = dates
    df_returns.set_index('Date', inplace=True)

    start_date, end_date = 3500, 3999
    df_ret = df_returns.loc[start_date:end_date]
    w = weights_df.reindex(df_ret.index, fill_value=0)

    notional = 1.0
    net_returns = []
    prev_w = None

    for date, row in df_ret.iterrows():
        curr_w = w.loc[date]
        gross_r = (curr_w * row).sum()
        if prev_w is None:
            turnover = np.abs(curr_w).sum()
        else:
            turnover = np.abs(curr_w - prev_w).sum()
        cost = cost_rate * turnover
        net_r = gross_r - cost
        net_returns.append(net_r)
        notional *= (1 + net_r)
        prev_w = curr_w

    net_return_pct = (notional - 1.0) * 100
    sharpe = np.mean(net_returns) / np.std(net_returns)
    return [net_return_pct, sharpe]


# -----------------------------------------------------------------------------
# TASK 1: FIVE BASE STRATEGIES
# -----------------------------------------------------------------------------

# load all data once
_train = pd.read_csv('train_data.csv')
_cv    = pd.read_csv('crossval_data.csv')
_all   = pd.concat([_train, _cv], ignore_index=True)
_close = _all.pivot(index='Date', columns='Symbol', values='Close')

def task1_Strategy1():
    days = range(3500, 4000)
    out = []
    for d in days:
        last_week = (d-1)//5
        if last_week < 50:
            out.append([0]*20)
            continue
        first_week = last_week - 49
        means = []
        for sym in range(20):
            wr = []
            for w in range(first_week, last_week+1):
                prev_c = 1.0 if w==1 else _close.iloc[5*(w-1), sym]
                cur_c  = _close.iloc[5*w, sym]
                wr.append((cur_c - prev_c)/prev_c)
            means.append(np.mean(wr))
        order = np.argsort(-np.array(means))
        w = np.zeros(20)
        tops = order[:6]; bots = order[-6:]
        w[tops] = -1/6; w[bots] = 1/6
        out.append(w)
    return pd.DataFrame(out, index=days, columns=range(20))


def task1_Strategy2():
    days = range(3500, 4000)
    out = []
    for d in days:
        e = d-1
        if e < 30:
            out.append([0]*20)
            continue
        sma5  = _close.iloc[e-5+1:e+1].mean(axis=0)
        lma30 = _close.iloc[e-30+1:e+1].mean(axis=0)
        rel   = (sma5 - lma30)/lma30
        order = np.argsort(-rel.values)
        w = np.zeros(20)
        w[order[:5]] = -1/5
        w[order[-5:]] =  1/5
        out.append(w)
    return pd.DataFrame(out, index=days, columns=range(20))


def task1_Strategy3():
    days = range(3500, 4000)
    out = []
    for d in days:
        if d-8 < 0:
            out.append([0]*20)
            continue
        p0 = _close.iloc[d-1]
        p7 = _close.iloc[d-8]
        roc = 100*(p0-p7)/p7
        order = np.argsort(-roc.values)
        w = np.zeros(20)
        w[order[:4]] = -1/4
        w[order[-4:]] =  1/4
        out.append(w)
    return pd.DataFrame(out, index=days, columns=range(20))


def task1_Strategy4():
    days = range(3500, 4000)
    out = []
    for d in days:
        e = d-1; s = e-21+1
        if s < 0:
            out.append([0]*20)
            continue
        win = _close.iloc[s:e+1]
        m21 = win.mean(axis=0); sd21 = win.std(axis=0)
        res = m21 + 3*sd21; sup = m21 - 3*sd21
        price = _close.iloc[e]
        prox_sup = (price - sup)/sup
        prox_res = (price - res)/res
        sup_ord = np.argsort(prox_sup.values)
        longs = sup_ord[:4]
        rem = [i for i in range(20) if i not in longs]
        res_ord = np.argsort(-prox_res.values[rem])
        shorts = [rem[i] for i in res_ord[:4]]
        w = np.zeros(20)
        w[longs]  =  1/4
        w[shorts] = -1/4
        out.append(w)
    return pd.DataFrame(out, index=days, columns=range(20))


def task1_Strategy5():
    days = range(3500, 4000)
    out = []
    for d in days:
        e = d-1; s = e-14+1
        if s < 0:
            out.append([0]*20)
            continue
        win = _close.iloc[s:e+1]
        high14 = win.max(axis=0); low14 = win.min(axis=0)
        price  = _close.iloc[e]
        rng    = high14 - low14
        k = pd.Series(0.0, index=range(20))
        for sym in range(20):
            if rng[sym] > 0:
                k[sym] = 100*(price[sym] - low14[sym])/rng[sym]
        order = np.argsort(k.values)
        w = np.zeros(20)
        w[order[:3]]  =  1/3
        w[order[-3:]] = -1/3
        out.append(w)
    return pd.DataFrame(out, index=days, columns=range(20))


def task1():
    S1 = task1_Strategy1()
    S2 = task1_Strategy2()
    S3 = task1_Strategy3()
    S4 = task1_Strategy4()
    S5 = task1_Strategy5()
    perf = {
        'Strategy1': backtester_without_TC(S1),
        'Strategy2': backtester_without_TC(S2),
        'Strategy3': backtester_without_TC(S3),
        'Strategy4': backtester_without_TC(S4),
        'Strategy5': backtester_without_TC(S5),
    }
    df = pd.DataFrame(perf, index=['NetReturn%','Sharpe']).T
    df.to_csv('task1.csv')
    print("Task1 performance:\n", df)
    return


# ---------------------------------------------------------------------
# TASK 2: ENSEMBLE WITHOUT TRANSACTION COSTS
# (identical to your last best‐performing version)
# ---------------------------------------------------------------------
def task2():
    import numpy as np
    import pandas as pd

    # ——— Load base strategies ———
    S1 = task1_Strategy1()
    S2 = task1_Strategy2()
    S3 = task1_Strategy3()
    S4 = task1_Strategy4()
    S5 = task1_Strategy5()
    bases = [S1, S2, S3, S4, S5]

    # ——— Load & pivot CV price data ———
    cv    = pd.read_csv('crossval_data.csv')
    price = cv.pivot(index='Date', columns='Symbol', values='Close')
    ret_df = price.pct_change().fillna(0).loc[3500:3999]
    days   = list(ret_df.index)
    ret    = ret_df.values

    # ——— Compute each strategy’s daily P&L over CV ———
    strat_ret = np.column_stack([
        (b.loc[days].values * ret).sum(axis=1)
        for b in bases
    ])  # shape (500×5)

    # ——— Regime indicators (30d momentum & dispersion) ———
    mom30  = ret_df.mean(axis=1).rolling(30).mean().shift(1)
    disp30 = ret_df.std(axis=1).rolling(30).mean().shift(1)
    th_mom  = mom30.median()
    th_disp = disp30.median()

    # ——— Raw regime labels ———
    raw_reg = pd.Series(index=days, dtype=int)
    for day in days:
        m, d = mom30.loc[day], disp30.loc[day]
        if   (m < th_mom) and (d < th_disp):   raw_reg[day] = 0
        elif (m < th_mom) and (d >= th_disp):  raw_reg[day] = 1
        elif (m >= th_mom) and (d < th_disp):  raw_reg[day] = 2
        else:                                  raw_reg[day] = 3

    # ——— Smooth regimes via rolling‐mode ———
    def rolling_mode(arr):
        vals, counts = np.unique(arr, return_counts=True)
        return vals[np.argmax(counts)]
    smooth_reg = raw_reg.rolling(window=5, min_periods=1).apply(rolling_mode, raw=True).astype(int)

    # ——— Precompute avg P&L by regime for each strategy ———
    avg_by_reg = {}
    for r in range(4):
        idxs = [i for i, day in enumerate(days) if raw_reg[day] == r]
        avg_by_reg[r] = strat_ret[idxs].mean(axis=0) if idxs else np.zeros(5)

    # ——— Build ensemble ———
    ensemble = pd.DataFrame(index=days, columns=price.columns, dtype=float)
    blend_map = {0:0.90, 1:0.80, 2:0.70, 3:0.60}

    for i, day in enumerate(days):
        # warm-up
        if i < 30:
            ensemble.loc[day] = S2.loc[day].values
            continue

        r = smooth_reg.loc[day]
        champ = int(np.argmax(avg_by_reg[r]))
        w_best = bases[champ].loc[day].values
        w2     = S2.loc[day].values

        α = blend_map[r]
        ensemble.loc[day] = α * w_best + (1 - α) * w2

    # ——— Save & evaluate ———
    ensemble.to_csv('task2_weights.csv')
    perf = backtester_without_TC(ensemble)
    pd.DataFrame({
        'NetReturn%': [perf[0]],
        'Sharpe':     [perf[1]]
    }).to_csv('task_2.csv', index=False)

    print("Task2 performance:", perf)
    return

# -----------------------------------------------------------------------------
# TASK 3: ULTRA–RISK-AVERSE ENSEMBLE
# -----------------------------------------------------------------------------
def task3():
    import numpy as np
    import pandas as pd

    # ——— Base strategy 2 only ———
    S2 = task1_Strategy2()

    # ——— Cross-val returns (3500–3999) ———
    cv    = pd.read_csv('crossval_data.csv')
    price = cv.pivot(index='Date', columns='Symbol', values='Close')
    ret_df = price.pct_change().fillna(0).loc[3500:3999]
    days   = list(ret_df.index)
    R      = ret_df.values

    # ——— Precompute Strategy2 daily P&L ———
    pnl2 = (S2.loc[days].values * R).sum(axis=1)
    pnl_series = pd.Series(pnl2, index=days)

    # ——— Rolling vol of Strategy2 (20-day window) ———
    rolling_vol = pnl_series.rolling(20).std().shift(1)

    # ——— Parameters ———
    target_vol     = 0.02  # desired daily vol
    loss_threshold = 0.02   # 2% one-day loss triggers flat
    max_leverage   = 1.5    # never lever >150%

    # ——— Build ensemble ———
    ensemble = pd.DataFrame(0.0, index=days, columns=price.columns, dtype=float)
    prev_pnl = pnl_series.iloc[0]

    for i, day in enumerate(days):
        base_w = S2.loc[day].values

        # 1) STOP-LOSS: if yesterday lost more than threshold → flat today
        if i>0 and (pnl_series.iloc[i-1] < -loss_threshold):
            w = np.zeros_like(base_w)
        else:
            # 2) VOL-TARGET: shrink exposure when vol is high
            vol = rolling_vol.loc[day]
            if np.isnan(vol) or vol < 1e-6:
                scale = 1.0
            else:
                scale = min(target_vol / vol, max_leverage)
            w = scale * base_w

        ensemble.loc[day] = w

    # ——— Save & backtest with transaction costs ———
    ensemble.to_csv('task3_weights.csv')
    net, shr = backtester_with_TC(ensemble, cost_rate=0.001)  # assume 0.1% per turnover unit
    pd.DataFrame({'NetReturn%':[net],'Sharpe':[shr]}).to_csv('task_3.csv', index=False)

    print("Task3 performance:", [net, shr])
    return
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    task1()
    task2()
    task3()