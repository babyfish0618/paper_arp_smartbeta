"""
单元测试
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.methods.equal_weight import equal_weight
from src.methods.risk_parity import risk_parity
from src.methods.agnostic_risk_parity import agnostic_risk_parity
from src.methods.eigen_risk_parity import eigen_risk_parity
from src.metrics import calculate_sharpe, calculate_max_drawdown


class TestMethods:
    """测试各种方法"""
    
    @pytest.fixture
    def sample_cov(self):
        """样本协方差矩阵"""
        return np.array([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.3.2, ],
            [00.3, 1.0]
        ])
    
    def test_equal_weight(self, sample_cov):
        """测试等权"""
        weights = equal_weight(sample_cov)
        assert len(weights) == 3
        assert np.abs(weights.sum() - 1.0) < 1e-6
        assert np.allclose(weights, [1/3, 1/3, 1/3])
    
    def test_risk_parity(self, sample_cov):
        """测试风险平价"""
        weights = risk_parity(sample_cov)
        assert len(weights) == 3
        assert np.abs(weights.sum() - 1.0) < 1e-6
        # 风险平价权重应该不等
        assert not np.allclose(weights, [1/3, 1/3, 1/3])
    
    def test_agnostic_risk_parity(self, sample_cov):
        """测试Agnostic Risk Parity"""
        weights = agnostic_risk_parity(sample_cov)
        assert len(weights) == 3
        assert np.abs(weights.sum() - 1.0) < 1e-3
    
    def test_eigen_risk_parity(self, sample_cov):
        """测试特征风险平价"""
        weights = eigen_risk_parity(sample_cov)
        assert len(weights) == 3
        assert np.abs(weights.sum() - 1.0) < 1e-3


class TestMetrics:
    """测试绩效指标"""
    
    @pytest.fixture
    def sample_returns(self):
        """样本收益率"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='B')
        return pd.Series(np.random.randn(252) * 0.02, index=dates)
    
    def test_sharpe(self, sample_returns):
        """测试夏普比率"""
        sharpe = calculate_sharpe(sample_returns)
        assert isinstance(sharpe, (int, float))
        assert not np.isnan(sharpe)
    
    def test_max_drawdown(self, sample_returns):
        """测试最大回撤"""
        mdd = calculate_max_drawdown(sample_returns)
        assert isinstance(mdd, (int, float))
        assert mdd <= 0  # 回撤应为负数


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
