"""
模拟数据生成模块

使用多因子模型生成股票收益率数据
"""

import numpy as np
import pandas as pd
from pathlib import Path


class DataGenerator:
    """多因子模型数据生成器"""
    
    def __init__(self, config: dict):
        """
        初始化
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.sim_config = config.get('simulation', {})
        self.n_stocks = self.sim_config.get('n_stocks', 500)
        self.n_days = self.sim_config.get('n_days', 252)
        self.n_factors = self.sim_config.get('n_factors', 5)
        self.factor_names = self.sim_config.get('factor_names', 
            ['F1', 'F2', 'F3', 'F4', 'F5'])
        
    def generate_factor_returns(self) -> np.ndarray:
        """
        生成因子收益率序列
        
        Returns:
            因子收益率矩阵 (n_days, n_factors)
        """
        mean = self.sim_config.get('factor_params', {}).get('mean', 0.0002)
        std = self.sim_config.get('factor_params', {}).get('std', 0.01)
        
        np.random.seed(42)
        return np.random.normal(mean, std, (self.n_days, self.n_factors))
    
    def generate_factor_exposures(self) -> np.ndarray:
        """
        生成因子暴露矩阵
        
        Returns:
            因子暴露矩阵 (n_stocks, n_factors)
        """
        min_exp = self.sim_config.get('exposure_params', {}).get('min_exposure', 0.0)
        max_exp = self.sim_config.get('exposure_params', {}).get('max_exposure', 1.0)
        
        np.random.seed(123)
        return np.random.uniform(min_exp, max_exp, (self.n_stocks, self.n_factors))
    
    def generate_idiosyncratic_returns(self) -> np.ndarray:
        """
        生成特异收益率
        
        Returns:
            特异收益率矩阵 (n_days, n_stocks)
        """
        mean = self.sim_config.get('idio_params', {}).get('mean', 0.0)
        std = self.sim_config.get('idio_params', {}).get('std', 0.02)
        
        np.random.seed(456)
        return np.random.normal(mean, std, (self.n_days, self.n_stocks))
    
    def generate_stock_returns(self) -> pd.DataFrame:
        """
        生成股票收益率
        
        Returns:
            股票收益率 DataFrame (n_days, n_stocks)
        """
        # 获取因子收益率和暴露
        factor_returns = self.generate_factor_returns()  # (n_days, n_factors)
        factor_exposures = self.generate_factor_exposures()  # (n_stocks, n_factors)
        idio_returns = self.generate_idiosyncratic_returns()  # (n_days, n_stocks)
        
        # 计算股票收益: r = X * f + epsilon
        # X: (n_stocks, n_factors), f: (n_days, n_factors) -> (n_days, n_stocks)
        stock_returns = factor_returns @ factor_exposures.T + idio_returns
        
        # 转为DataFrame
        stock_ids = [f"STOCK_{i:04d}" for i in range(self.n_stocks)]
        dates = pd.date_range(start='2020-01-02', periods=self.n_days, freq='B')
        
        return pd.DataFrame(stock_returns, index=dates, columns=stock_ids)
    
    def generate_all_data(self) -> dict:
        """
        生成所有需要的数据
        
        Returns:
            包含所有数据的字典
        """
        return {
            'stock_returns': self.generate_stock_returns(),
            'factor_returns': self.generate_factor_returns(),
            'factor_exposures': self.generate_factor_exposures(),
            'factor_names': self.factor_names
        }
    
    def save_data(self, output_dir: str = 'data'):
        """
        保存生成的数据到CSV
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data = self.generate_all_data()
        
        # 保存股票收益率
        data['stock_returns'].to_csv(output_path / 'stock_returns.csv')
        
        # 保存因子收益率
        factor_ret_df = pd.DataFrame(
            data['factor_returns'],
            columns=self.factor_names,
            index=pd.date_range(start='2020-01-02', periods=self.n_days, freq='B')
        )
        factor_ret_df.to_csv(output_path / 'factor_returns.csv')
        
        # 保存因子暴露
        exposure_df = pd.DataFrame(
            data['factor_exposures'],
            columns=self.factor_names,
            index=[f"STOCK_{i:04d}" for i in range(self.n_stocks)]
        )
        exposure_df.to_csv(output_path / 'factor_exposures.csv')
        
        print(f"数据已保存到 {output_dir}/")
        return data


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """加载配置文件"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == '__main__':
    # 测试
    config = load_config('../config/config.yaml')
    generator = DataGenerator(config)
    data = generator.save_data('../data')
    print("股票收益率 shape:", data['stock_returns'].shape)
    print("因子暴露 shape:", data['factor_exposures'].shape)
