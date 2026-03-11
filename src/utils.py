"""
工具函数模块
"""

import numpy as np
import pandas as pd
from pathlib import Path


def ensure_dir(path: str) -> Path:
    """
    确保目录存在
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str) -> dict:
    """加载YAML配置"""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    """保存YAML配置"""
    import yaml
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    归一化权重
    
    Args:
        weights: 权重数组
        
    Returns:
        归一化后的权重
    """
    return weights / weights.sum()


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    计算组合波动率
    
    Args:
        weights: 权重数组
        cov_matrix: 协方差矩阵
        
    Returns:
        组合波动率
    """
    return np.sqrt(weights @ cov_matrix @ weights)


def correlation_to_covariance(corr_matrix: np.ndarray, 
                             vols: np.ndarray) -> np.ndarray:
    """
    从相关矩阵和波动率构建协方差矩阵
    
    Args:
        corr_matrix: 相关矩阵
        vols: 波动率数组
        
    Returns:
        协方差矩阵
    """
    return corr_matrix * np.outer(vols, vols)


def annualized_factor(factor_ret: pd.Series, trading_days: int = 240) -> pd.Series:
    """
    年化因子收益
    
    Args:
        factor_ret: 因子收益序列
        trading_days: 年化交易日数
        
    Returns:
        年化因子收益
    """
    return factor_ret * np.sqrt(trading_days)


__all__ = [
    'ensure_dir',
    'load_yaml',
    'save_yaml',
    'normalize_weights',
    'portfolio_volatility',
    'correlation_to_covariance',
    'annualized_factor'
]
