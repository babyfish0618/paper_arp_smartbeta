"""
Smart Beta 构建模块

基于因子暴露选取Top N股票构建Smart Beta组合
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class SmartBetaBuilder:
    """Smart Beta组合构建器"""
    
    def __init__(self, config: dict):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.smartbeta_config = config.get('smartbeta', {})
        self.top_n = self.smartbeta_config.get('top_n', 20)
        self.equal_weight = self.smartbeta_config.get('equal_weight', True)
    
    def select_top_stocks(self, 
                         factor_exposures: np.ndarray,
                         factor_names: List[str],
                         factor_name: str) -> Tuple[List[int], np.ndarray]:
        """
        选取因子暴露最高的Top N股票
        
        Args:
            factor_exposures: 因子暴露矩阵 (n_stocks, n_factors)
            factor_names: 因子名称列表
            factor_name: 目标因子名称
            
        Returns:
            (股票索引列表, 因子暴露值数组)
        """
        # 找到因子索引
        if factor_name in factor_names:
            factor_idx = factor_names.index(factor_name)
        else:
            factor_idx = factor_names.index(factor_names[0])  # 默认第一个
        
        # 获取该因子的暴露
        exposures = factor_exposures[:, factor_idx]
        
        # 排序并选取Top N
        top_indices = np.argsort(exposures)[-self.top_n:][::-1]
        top_exposures = exposures[top_indices]
        
        return top_indices.tolist(), top_exposures
    
    def build_factor_portfolio(self,
                              factor_exposures: np.ndarray,
                              factor_names: List[str],
                              factor_name: str,
                              stock_returns: pd.DataFrame) -> pd.Series:
        """
        构建因子组合
        
        Args:
            factor_exposures: 因子暴露矩阵
            factor_names: 因子名称列表
            factor_name: 目标因子名
            stock_returns: 股票收益率 DataFrame
            
        Returns:
            组合日收益率序列
        """
        # 选取Top N股票
        top_indices, _ = self.select_top_stocks(
            factor_exposures, factor_names, factor_name
        )
        
        # 获取这些股票的收益率
        selected_returns = stock_returns.iloc[:, top_indices]
        
        # 等权计算组合收益
        portfolio_returns = selected_returns.mean(axis=1)
        
        return portfolio_returns
    
    def build_all_factor_portfolios(self,
                                   factor_exposures: np.ndarray,
                                   factor_names: List[str],
                                   stock_returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        构建所有因子的Smart Beta组合
        
        Args:
            factor_exposures: 因子暴露矩阵
            factor_names: 因子名称列表
            stock_returns: 股票收益率
            
        Returns:
            因子组合收益率字典 {因子名: 收益率序列}
        """
        portfolios = {}
        
        for factor_name in factor_names:
            print(f"构建 {factor_name} 因子组合...")
            portfolios[factor_name] = self.build_factor_portfolio(
                factor_exposures, factor_names, factor_name, stock_returns
            )
        
        return portfolios
    
    def get_portfolio_stats(self,
                           factor_exposures: np.ndarray,
                           factor_names: List[str],
                           factor_name: str) -> dict:
        """
        获取组合统计信息
        
        Args:
            factor_exposures: 因子暴露矩阵
            factor_names: 因子名称列表
            factor_name: 目标因子
            
        Returns:
            组合统计信息
        """
        top_indices, top_exposures = self.select_top_stocks(
            factor_exposures, factor_names, factor_name
        )
        
        return {
            'factor_name': factor_name,
            'n_stocks': len(top_indices),
            'stock_indices': top_indices,
            'mean_exposure': np.mean(top_exposures),
            'total_exposure': np.sum(top_exposures)
        }


def load_factor_exposures(file_path: str) -> Tuple[np.ndarray, List[str]]:
    """加载因子暴露数据"""
    df = pd.read_csv(file_path, index_col=0)
    return df.values, df.columns.tolist()


def load_stock_returns(file_path: str) -> pd.DataFrame:
    """加载股票收益率数据"""
    return pd.read_csv(file_path, index_col=0, parse_dates=True)


if __name__ == '__main__':
    # 测试
    import yaml
    
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    builder = SmartBetaBuilder(config)
    print(f"Smart Beta Builder 初始化完成")
    print(f"Top N: {builder.top_n}")
    print(f"等权: {builder.equal_weight}")
