# Quantitative Smart Beta Portfolio Project

量化Smart Beta投资组合项目 - 支持 Agnostic Risk Parity 策略

## 项目结构

```
quant_smartbeta/
├── config/
│   └── config.yaml          # 配置文件 (参数设置)
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # 模拟数据生成模块
│   ├── smartbeta.py         # Smart Beta构建模块
│   ├── methods/             # 各种组合优化方法
│   │   ├── __init__.py
│   │   ├── equal_weight.py     # 等权
│   │   ├── risk_parity.py      # 传统风险平价
│   │   ├── agnostic_risk_parity.py  # Agnostic Risk Parity
│   │   └── eigen_risk_parity.py # 特征风险平价
│   ├── backtest.py         # 回测模块
│   ├── metrics.py          # 绩效指标计算
│   └── utils.py            # 工具函数
├── data/
│   └── .gitkeep
├── notebooks/
│   └── demo.ipynb           # 示例notebook
├── tests/
│   └── test_methods.py      # 单元测试
├── requirements.txt
├── README.md
└── main.py                 # 主程序入口
```

## 模块说明

### 1. config/config.yaml
配置文件，包含所有可调参数：
- 股票数量、时间天数、因子数量等模拟参数
- 因子参数（均值、方差等）
- 回测参数

### 2. src/data_generator.py
模拟数据生成模块：
- 多因子模型生成股票收益率
- 生成因子暴露矩阵
- 支持自定义参数

### 3. src/smartbeta.py
Smart Beta构建模块：
- 基于因子暴露选取Top N股票
- 构建因子组合

### 4. src/methods/
各种组合优化方法：
- equal_weight: 等权重
- risk_parity: 传统风险平价
- agnostic_risk_parity: Agnostic Risk Parity (论文方法)
- eigen_risk_parity: 特征风险平价

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行示例
python main.py

# 运行测试
pytest tests/
```
