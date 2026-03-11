"""
特征风险平价 (Eigen Risk Parity)
基于PCA特征值的风险平价方法
"""

import numpy as np
import pandas as pd
from typing import Optional


def eigen_risk_parity(cov_matrix: np.ndarray, 
                      max_iter: int = 1000,
                      tolerance: float = 1e-8) -> np.ndarray:
    """
    特征风险平价权重
    
    基于协方差矩阵的特征值分解，将风险分配到各个特征向量方向
    
    Args:
        cov_matrix: 协方差矩阵 (n x n)
        max_iter: 最大迭代次数
        tolerance: 收敛容忍度
        
    Returns:
        权重向量 (n,)
    """
    n = cov_matrix.shape[0]
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 按特征值降序排列
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 计算特征风险贡献
    # 每个主成分的风险等于其特征值的平方根
    eigen_risks = np.sqrt(np.maximum(eigenvalues, 0))
    
    # 风险平价目标：每个特征方向风险相等
    # 使用迭代方法求解
    weights = np.ones(n) / n
    
    for _ in range(max_iter):
        # 计算当前权重对应的风险
        portfolio_variance = weights @ cov_matrix @ weights
        marginal_risk = cov_matrix @ weights
        risk_contribution = weights * marginal_risk / np.sqrt(portfolio_variance)
        
        # 目标：使风险贡献相等
        target_risk = np.mean(risk_contribution)
        
        # 更新权重
        new_weights = np.where(
            marginal_risk > 0,
            target_risk / marginal_risk,
            weights
        )
        new_weights = new_weights / new_weights.sum()
        
        # 检查收敛
        if np.abs(new_weights - weights).max() < tolerance:
            break
            
        weights = new_weights
        
    return weights


def eigen_risk_parity_from_returns(returns: pd.DataFrame,
                                    n_components: Optional[int] = None) -> pd.Series:
    """
    从收益率数据计算特征风险平价权重
    
    Args:
        returns: 收益率数据 (n_samples x n_assets)
        n_components: 主成分数量，None表示全部
        
    Returns:
        权重 Series
    """
    # 计算协方差矩阵
    cov_matrix = returns.cov().values
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 选择主成分
    if n_components is not None:
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 重建降维后的协方差矩阵
        cov_reduced = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        cov_matrix = cov_reduced
    
    # 计算权重
    weights = eigen_risk_parity(cov_matrix)
    
    return pd.Series(weights, index=returns.columns)


class EigenRiskParity:
    """特征风险平价策略类"""
    
    def __init__(self, n_components: Optional[int] = None):
        """
        初始化
        
        Args:
            n_components: 主成分数量，None表示全部
        """
        self.n_components = n_components
        self.weights = None
        
    def fit(self, returns: pd.DataFrame) -> 'EigenRiskParity':
        """
        拟合模型
        
        Args:
            returns: 历史收益率数据
            
        Returns:
            self
        """
        self.weights = eigen_risk_parity_from_returns(returns, self.n_components)
        return self
        
    def get_weights(self) -> pd.Series:
        """获取权重"""
        if self.weights is None:
            raise ValueError("Model not fitted yet")
        return self.weights
    
    def predict(self, returns: pd.DataFrame) -> np.ndarray:
        """
        预测权重
        
        Args:
            returns: 当前收益率数据
            
        Returns:
            权重向量
        """
        if self.weights is None:
            self.fit(returns)
        return self.weights.values
