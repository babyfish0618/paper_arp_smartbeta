"""
量化Smart Beta组合 - 主程序入口
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from data_generator import DataGenerator
from smartbeta import SmartBetaBuilder
from methods.equal_weight import EqualWeightPortfolio
from methods.risk_parity import RiskParityPortfolio
from methods.agnostic_risk_parity import AgnosticRiskParityPortfolio
from metrics import calculate_all_metrics


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """加载配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """主函数"""
    print("="*60)
    print("量化Smart Beta组合项目")
    print("="*60)
    
    # 加载配置
    config = load_config('config/config.yaml')
    print(f"\n配置加载完成")
    print(f"  股票数量: {config['simulation']['n_stocks']}")
    print(f"  交易天数: {config['simulation']['n_days']}")
    print(f"  因子数量: {config['simulation']['n_factors']}")
    
    # 1. 生成模拟数据
    print("\n[1] 生成模拟数据...")
    generator = DataGenerator(config)
    data = generator.generate_all_data()
    print(f"  股票收益率 shape: {data['stock_returns'].shape}")
    print(f"  因子暴露 shape: {data['factor_exposures'].shape}")
    
    # 保存数据
    generator.save_data('data')
    
    # 2. 构建Smart Beta组合
    print("\n[2] 构建Smart Beta组合...")
    builder = SmartBetaBuilder(config)
    
    # 获取第一个因子的组合
    factor_name = config['simulation']['factor_names'][0]
    print(f"  构建 {factor_name} 因子组合...")
    
    # 选取Top N股票
    top_indices, _ = builder.select_top_stocks(
        data['factor_exposures'],
        data['factor_names'],
        factor_name
    )
    print(f"  选取股票数: {len(top_indices)}")
    
    # 获取这些股票的收益率
    selected_returns = data['stock_returns'].iloc[:, top_indices]
    print(f"  组合收益率 shape: {selected_returns.shape}")
    
    # 3. 计算不同方法的权重
    print("\n[3] 计算组合权重...")
    
    methods = {
        '等权': EqualWeightPortfolio(config),
        '风险平价': RiskParityPortfolio(config),
        'AGNOSTIC风险平价': AgnosticRiskParityPortfolio(config)
    }
    
    results = {}
    trading_days = config['portfolio']['trading_days']
    rf = config['portfolio']['risk_free_rate']
    
    for name, method in methods.items():
        print(f"  计算 {name}...")
        try:
            weights = method.calculate_weights(selected_returns)
            portfolio_returns = selected_returns @ weights
            metrics = calculate_all_metrics(portfolio_returns, trading_days, rf)
            results[name] = {
                'weights': weights,
                'metrics': metrics
            }
            print(f"    年化收益率: {metrics['年化收益率']:.2%}")
            print(f"    夏普比率: {metrics['夏普比率']:.4f}")
        except Exception as e:
            print(f"    错误: {e}")
    
    # 4. 输出结果
    print("\n[4] 绩效对比")
    print("-"*60)
    for name, result in results.items():
        print(f"\n{name}:")
        for k, v in result['metrics'].items():
            if isinstance(v, float):
                if abs(v) < 10:
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)


if __name__ == '__main__':
    main()
