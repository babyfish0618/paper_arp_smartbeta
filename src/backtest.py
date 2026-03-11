"""
回测模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class Backtest:
    """回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000):
        """
        初始化回测
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        
    def buy(self, date, symbol: str, price: float, quantity: int):
        """买入"""
        cost = price * quantity * 1.001  # 考虑手续费
        if self.capital >= cost:
            self.capital -= cost
            if symbol not in self.positions:
                self.positions[symbol] = {'quantity': 0, 'cost': 0}
            self.positions[symbol]['quantity'] += quantity
            self.positions[symbol]['cost'] += cost
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'buy',
                'price': price,
                'quantity': quantity
            })
            return True
        return False
    
    def sell(self, date, symbol: str, price: float, quantity: int):
        """卖出"""
        if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
            revenue = price * quantity * 0.999
            self.capital += revenue
            self.positions[symbol]['quantity'] -= quantity
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'sell',
                'price': price,
                'quantity': quantity
            })
            return True
        return False
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """计算组合市值"""
        value = self.capital
        for symbol, pos in self.positions.items():
            if symbol in prices and pos['quantity'] > 0:
                value += prices[symbol] * pos['quantity']
        return value
    
    def run(self, 
            signals: pd.DataFrame, 
            prices: pd.DataFrame,
            rebalance_freq: str = 'M') -> Dict:
        """
        运行回测
        
        Args:
            signals: 信号数据，index为日期，columns为标的
            prices: 价格数据
            rebalance_freq: 再平衡频率 ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            回测结果字典
        """
        self.reset()
        
        # 按月/季/年再平衡
        if rebalance_freq == 'M':
            rebalance_dates = signals.resample('ME').first().dropna().index
        elif rebalance_freq == 'Q':
            rebalance_dates = signals.resample('QE').first().dropna().index
        elif rebalance_freq == 'Y':
            rebalance_dates = signals.resample('YE').first().dropna().index
        else:
            rebalance_dates = signals.index
            
        for date in signals.index:
            if date in prices.index:
                # 获取当前价格
                current_prices = prices.loc[date].to_dict()
                current_value = self.get_portfolio_value(current_prices)
                self.portfolio_values.append(current_value)
                self.dates.append(date)
                
                # 再平衡
                if date in rebalance_dates:
                    target_weights = signals.loc[date].to_dict()
                    # 这里实现再平衡逻辑
                    pass
                    
        return self.get_results()
    
    def get_results(self) -> Dict:
        """获取回测结果"""
        if not self.portfolio_values:
            return {}
            
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        total_return = (values[-1] - values[0]) / values[0]
        annual_return = (1 + total_return) ** (252 / len(values)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': values[-1],
            'num_trades': len(self.trades)
        }
