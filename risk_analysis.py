import numpy as np

def calculate_risk(prices):

    # FIX: force to 1D numpy array regardless of what shape comes in
    prices = np.array(prices).flatten()

    # FIX: need at least 2 prices to compute returns
    if len(prices) < 2:
        return "Unknown", 0.0

    returns = np.diff(prices) / prices[:-1]

    # FIX: explicitly flatten returns too, then take scalar std
    volatility = float(np.std(returns.flatten())) * 100

    if volatility < 1.2:
        risk_level = "Low Risk"
    elif volatility < 2.2:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    return risk_level, volatility