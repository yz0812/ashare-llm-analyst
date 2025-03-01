# AStock-LLM-Analyst

一个基于Python的A股智能分析工具，结合大语言模型提供数据驱动的投资建议和市场洞察。

## 项目简介

AStock-LLM-Analyst 是一个A股市场的技术分析工具，通过[Ashare](https://github.com/mpquant/Ashare)采集股票历史数据，[MyTT](https://github.com/mpquant/MyTT)计算常见技术指标（如MACD、KDJ、RSI等），并利用大语言模型（Deepseek）生成可读性强的投资建议和市场分析。

该工具能够自动生成完整的HTML分析报告，包括基础数据分析、技术指标计算、趋势判断、支撑/阻力位识别以及AI辅助的专业投资建议。

## 主要功能

- 自动获取A股历史交易数据
- 计算超过25种技术指标（MA、MACD、KDJ、RSI、BOLL等）
- 生成详细的技术分析图表
- 使用Deepseek大语言模型提供专业的投资分析和建议
- 输出美观的HTML格式分析报告

## 使用方法

### 前置准备

1. 确保安装了所有必需的依赖项:
```bash
pip install pandas numpy matplotlib pytz
```

2. **重要**: 在使用前，需要将`main.py`中的`{Replace with your Key}`替换为你自己的Deepseek API密钥：
```python
analyzer = StockAnalyzer(stock_info, deepseek_api_key='{Replace with your Key}')
```

### 运行分析

1. 在`main.py`中设置要分析的股票代码：
```python
stock_info = {
    '股票名称': '股票代码',  # 例如 '上证指数': 'sh000001'
}
```

2. 运行主程序：
```bash
python main.py
```

3. 分析报告将自动生成并保存在`public/index.html`路径下

## 技术架构

- 数据获取：使用Ashare模块获取A股历史数据
- 技术分析：使用MyTT库进行技术指标计算
- 图表生成：使用Matplotlib生成技术分析图表
- AI分析：通过Deepseek API获取专业的投资建议
- 报告生成：生成包含详细分析的HTML报告

## 输出示例

生成的分析报告包含以下内容：

1. 基础技术分析（收盘价、涨跌幅、成交量等）
2. 技术指标详情（各项指标的最新值）
3. 技术指标图表（多维度的股票走势分析图）
4. 人工智能分析报告（基于历史数据的专业分析和投资建议）

## 重要说明

- **安全提示**：该项目是由个人自用的私有仓库公开而来，API凭据的存储并未做特别的安全防范措施。请务必妥善保管你的API密钥，建议使用环境变量或配置文件来存储敏感信息。

- **输出位置**：分析结果会输出到根目录下的`public`文件夹中。如果文件夹不存在，程序会自动创建。

- **免责声明**：本工具仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。用户应对自己的投资决策负责。

## 后续开发计划

- 添加更多技术指标和分析维度
- 支持批量分析多只股票
- 提供更丰富的可视化选项
- 增加历史数据对比和回测功能
- 优化AI分析模型和提示词设计

## 许可证

[MIT License](LICENSE)



## docker 部署
docker build -t ashare-llm-analyst .
