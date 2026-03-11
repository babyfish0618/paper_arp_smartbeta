"""
Agnostic Risk Parity 方法

论文: Agnostic Risk Parity: Taming Known and Unknown-Unknowns
作者: Benichou et al., 2016

核心思想: 基于对称性论证，不依赖历史协方差估计
"""

import numpy as np
import pandas as pd
from scipy.linalg import eigh


class AgnosticRiskParityPortfolio:
    """
    Agnostic Risk Parity (AGP) 组合
    
    核心特点:
    - 不直接使用协方差矩阵
    - 通过特征值分解处理风险
    - 使每个特征向量对组合风险的贡献相等
    """
    
    def __init__(self, config: dict = None):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
    
    def calculate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """计算协方差矩阵"""
        return returns.cov().values
    
    def calculate_correlation(self, returns: pd.DataFrame) -> np.ndarray:
        """计算相关矩阵"""
        return returns.corr().values
    
    def eigen_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        特征风险平价 (Eigen Risk Parity)
        
        使得每个特征向量对组合风险的贡献相等
        
        Args:
            cov_matrix: 协方差矩阵
            
        Returns:
            权重数组
        """
        # 特征值分解
        eigenvalues, eigenvectors = eigh(cov_matrix)
        
        # 只保留正特征值
        idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        n_eigen = len(eigenvalues)
        
        # 目标：每个特征向量的风险贡献相等
        # 即每个特征值对应的权重贡献相等
        target_contrib = np.ones(n_eigen) / n_eigen
        
        # 求解权重
        # 组合方差 = sum(w^2 * lambda_i)
        # 特征向量风险贡献 = w^2 * lambda_i / 方差
        
        def objective(w):
            exp_var = eigenvalues @ (w ** 2)
            contrib = (w ** 2) * eigenvalues / (exp_var + 1e-10)
            return np.sum((contrib - target_contrib) ** 2)
        
        # 初始权重
        n_assets = cov_matrix.shape[0]
        x0 = np.ones(n_assets) / n_assets
        
        # 约束
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # 优化
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def agnostic_portfolio(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Agnostic Risk Parity 组合
        
        方法:
        1. 计算相关矩阵
        2. 进行特征值分解
        3. 基于对称性分配权重
        
        Args:
            returns: 资产收益率 DataFrame
            
        Returns:
            权重数组
        """
        corr_matrix = self.calculate_correlation(returns)
        
        # 标准化协方差矩阵 (即相关矩阵)
        # 简化版AGP: 基于相关矩阵的特征分解
        return self.eigen_risk_parity(corr_matrix)
    
    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """
        计算AGNOSTIC风险平价权重
        
        Args:
            returns: 资产收益率 DataFrame
            
        Returns:
            权重数组
        """
        return self.agnostic_portfolio(returns)
    
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
    
    def get_risk_contribution(self, 
                           weights: np.ndarray, 
                           returns: pd.DataFrame) -> np.ndarray:
        """
        计算各资产的风险贡献
        
        Args:
            weights: 权重数组
            returns: 资产收益率
            
        Returns:
            各资产的风险贡献
        """
        cov_matrix = self.calculate_covariance(returns)
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_risk = cov_matrix @ weights
        risk_contrib = weights * marginal_risk / portfolio_vol
        return risk_contrib


class EigenRiskParityPortfolio(AgnosticRiskParityPortfolio):
    """特征风险平价组合 (AGP的简化版本)"""
    
    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """计算特征风险平价权重"""
        cov_matrix = self.calculate_covariance(returns)
        return self.eigen_risk_parity(cov_matrix)


if __name__ == '__main__':
    # 测试
    np.random.seed(42)
    returns = pd.DataFrame(
        np.random.randn(100, 5) * 0.02,
        columns=['A', 'B', 'C', 'D', 'E']
    )
    
    # 生成相关性较强的数据
    returns = pd.DataFrame(
        np.random.randn(100, 5) @ np.array([
            [1.0, 0.5, 0.3, 0.1, 0.0],
            [0.5, 1.0, 0.4, 0.2, 0.1],
            [0.3, 0.4, 1.0, 0.3, 0.1],
            [0.1, 0.2, 0.3, 1.0, 0.2],
            [0.0, 0.1, 0.1, 0.2, 1.0]
        ]) * 0.02,
        columns=['A', 'B', 'C', 'D', 'E']
    )
    
    portfolio = AgnosticRiskParityPortfolio()
    weights = portfolio.calculate_weights(returns)
    print("AGNOSTIC风险平价权重:", weights)
