"""
传统风险平价组合方法

Risk Parity: 每个资产对组合风险的贡献相等
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParityPortfolio:
    """传统风险平价组合"""
    
    def __init__(self, config: dict = None):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
    
    def calculate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        计算协方差矩阵
        
        Args:
            returns: 资产收益率 DataFrame
            
        Returns:
            协方差矩阵
        """
        return returns.cov().values
    
    def risk_contribution(self, weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """
        计算各资产的风险贡献
        
        Args:
            weights: 权重数组
            cov_matrix: 协方差矩阵
            
        Returns:
            各资产的风险贡献
        """
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_risk = cov_matrix @ weights
        risk_contrib = weights * marginal_risk / portfolio_vol
        return risk_contrib
    
    def risk_parity_objective(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """
        风险平价目标函数
        
        使得各资产的风险贡献相等
        """
        n = len(weights)
        target_rc = np.ones(n) / n  # 目标风险贡献相等
        current_rc = self.risk_contribution(weights, cov_matrix)
        return np.sum((current_rc - target_rc) ** 2)
    
    def calculate_weights(self, 
                        returns: pd.DataFrame,
                        allow_short: bool = False) -> np.ndarray:
        """
        计算风险平价权重
        
        Args:
            returns: 资产收益率 DataFrame
            allow_short: 是否允许做空
            
        Returns:
            权重数组
        """
        cov_matrix = self.calculate_covariance(returns)
        n_assets = returns.shape[1]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # 边界
        if allow_short:
            bounds = [(-1, 1) for _ in range(n_assets)]
        else:
            bounds = [(0, 1) for _ in range(n_assets)]
        
        # 优化
        result = minimize(
            self.risk_parity_objective,
            x0,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def get_portfolio_returns(self,
                            returns: pd.DataFrame,
                            weights: np.ndarray = None) -> pd.Series:
        """
        计算组合收益
        
        Args:
            returns: 资产收益率 DataFrame
            weights: 权重数组
            
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
        np.random.randn(100, 5) * 0.02,
        columns=['A', 'B', 'C', 'D', 'E']
    )
    
    portfolio = RiskParityPortfolio()
    weights = portfolio.calculate_weights(returns)
    print("风险平价权重:", weights)
    print("风险贡献:", portfolio.risk_contribution(weights, portfolio.calculate_covariance(returns)))
