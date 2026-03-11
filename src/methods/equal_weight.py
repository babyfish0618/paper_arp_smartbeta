"""
等权重组合方法
"""

import numpy as np
import pandas as pd


class EqualWeightPortfolio:
    """等权重组合"""
    
    def __init__(self, config: dict = None):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
    
    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """
        计算等权重
        
        Args:
            returns: 资产收益率 DataFrame
            
        Returns:
            权重数组
        """
        n_assets = returns.shape[1]
        return np.ones(n_assets) / n_assets
    
    def get_portfolio_returns(self, 
                            returns: pd.DataFrame, 
                            weights: np.ndarray = None) -> pd.Series:
        """
        计算组合收益
        
        Args:
            returns: 资产收益率 DataFrame
            weights: 权重数组 (可选)
            
        Returns:
            组合收益率序列
        """
        if weights is None:
            weights = self.calculate_weights(returns)
        
        return returns @ weights


if __name__ == '__main__':
    # 测试
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(100, 5),
        columns=['A', 'B', 'C', 'D', 'E']
    )
    
    portfolio = EqualWeightPortfolio()
    weights = portfolio.calculate_weights(returns)
    print("等权重:", weights)
    print("组合收益:", portfolio.get_portfolio_returns(returns).head())
