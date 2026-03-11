"""
绩效指标计算模块
"""

import numpy as np
import pandas as pd


def calculate_cumulative_return(returns: pd.Series) -> float:
    """
    计算累计收益率
    
    Args:
        returns: 收益率序列
        
    Returns:
        累计收益率
    """
    return (1 + returns).prod() - 1


def calculate_annual_return(returns: pd.Series, trading_days: int = 240) -> float:
    """
    计算年化收益率
    
    Args:
        returns: 收益率序列
        trading_days: 年化交易日数
        
    Returns:
        年化收益率
    """
    cum_ret = calculate_cumulative_return(returns)
    n_days = len(returns)
    return (1 + cum_ret) ** (trading_days / n_days) - 1


def calculate_annual_volatility(returns: pd.Series, trading_days: int = 240) -> float:
    """
    计算年化波动率
    
    Args:
        returns: 收益率序列
        trading_days: 年化交易日数
        
    Returns:
        年化波动率
    """
    return returns.std() * np.sqrt(trading_days)


def calculate_sharpe_ratio(annual_return: float, 
                          annual_volatility: float, 
                          risk_free_rate: float = 0.02) -> float:
    """
    计算夏普比率
    
    Args:
        annual_return: 年化收益率
        annual_volatility: 年化波动率
        risk_free_rate: 无风险利率
        
    Returns:
        夏普比率
    """
    return (annual_return - risk_free_rate) / annual_volatility


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    计算最大回撤
    
    Args:
        returns: 收益率序列
        
    Returns:
        最大回撤 (负数)
    """
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()


def calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """
    计算卡玛比率
    
    Args:
        annual_return: 年化收益率
        max_drawdown: 最大回撤
        
    Returns:
        卡玛比率
    """
    return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0


def calculate_win_rate(returns: pd.Series) -> float:
    """
    计算胜率
    
    Args:
        returns: 收益率序列
        
    Returns:
        胜率 (0-1)
    """
    return (returns > 0).mean()


def calculate_monthly_win_rate(returns: pd.Series) -> float:
    """
    计算月均胜率
    
    Args:
        returns: 日收益率序列
        
    Returns:
        月均胜率
    """
    monthly_returns = returns.resample('ME').apply(lambda x: (1+x).prod() - 1)
    return (monthly_returns > 0).mean()


def calculate_all_metrics(returns: pd.Series, 
                        trading_days: int = 240,
                        risk_free_rate: float = 0.02) -> dict:
    """
    计算所有绩效指标
    
    Args:
        returns: 收益率序列
        trading_days: 年化交易日数
        risk_free_rate: 无风险利率
        
    Returns:
        包含所有指标的字典
    """
    cum_ret = calculate_cumulative_return(returns)
    annual_ret = calculate_annual_return(returns, trading_days)
    annual_vol = calculate_annual_volatility(returns, trading_days)
    sharpe = calculate_sharpe_ratio(annual_ret, annual_vol, risk_free_rate)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(annual_ret, max_dd)
    win_rate = calculate_win_rate(returns)
    monthly_win_rate = calculate_monthly_win_rate(returns)
    
    return {
        '累计收益率': cum_ret,
        '年化收益率': annual_ret,
        '年化波动率': annual_vol,
        '夏普比率': sharpe,
        '最大回撤': max_dd,
        '卡玛比率': calmar,
        '日胜率': win_rate,
        '月均胜率': monthly_win_rate
    }


if __name__ == '__main__':
    # 测试
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.02)
    
    metrics = calculate_all_metrics(returns)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
