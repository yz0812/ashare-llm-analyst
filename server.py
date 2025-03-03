from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from main import StockAnalyzer

app = Flask(__name__)
load_dotenv()  # 加载.env文件

# 验证环境变量加载
app.logger.info("正在验证环境变量加载...")
env_path = os.path.join(os.getcwd(), '.env')
app.logger.info(f"环境变量文件路径: {env_path}")
app.logger.info(f"环境变量文件是否存在: {os.path.exists(env_path)}")

# 读取并打印所有STOCK_开头的环境变量
stock_vars = {k: v for k, v in os.environ.items() if k.startswith('STOCK_')}
app.logger.info(f"加载的股票环境变量: {stock_vars}")

# 创建logs目录（如果不存在）
if not os.path.exists('logs'):
    os.makedirs('logs')

# 配置日志
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

# 配置文件处理器
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

# 配置控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

# 添加处理器到应用日志记录器
app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(logging.INFO)

# 读取配置项
config1 = os.getenv('CONFIG1')
config2 = os.getenv('CONFIG2')

app.logger.info(f"Config1: {config1}")
app.logger.info(f"Config2: {config2}")

@app.route('/')
def index():
    app.logger.info("访问首页")
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    app.logger.info(f"访问静态文件: {path}")
    return send_from_directory('static', path)

@app.route('/get_stocks', methods=['GET'])
def get_stocks():
    try:
        # 添加调试日志
        app.logger.info("开始获取股票信息")
        app.logger.info(f"当前所有环境变量: {dict(os.environ)}")
        
        stock_info = {}
        for key, value in os.environ.items():
            app.logger.info(f"检查环境变量: {key}")
            if key.startswith('STOCK_'):
                stock_name = key.replace('STOCK_', '')
                stock_info[stock_name] = value
                app.logger.info(f"找到股票信息: {stock_name} = {value}")
        
        app.logger.info(f"最终获取到的股票信息: {stock_info}")
        return jsonify(stock_info)
    except Exception as e:
        app.logger.error(f"获取股票信息失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_stocks', methods=['POST'])
def save_stocks():
    try:
        stock_info = request.json
        app.logger.info(f"保存股票信息: {stock_info}")
        
        env_content = []
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if not line.strip().startswith('STOCK_'):
                        env_content.append(line.strip())
        
        for name, code in stock_info.items():
            env_content.append(f'STOCK_{name}={code}')
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write('\n'.join(env_content))
        
        app.logger.info("股票信息保存成功")
        return jsonify({"status": "success"})
    
    except Exception as e:
        app.logger.error(f"保存股票信息失败: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/analyze_stocks', methods=['POST'])
def analyze_stocks():
    try:
        selected_stocks = request.json
        app.logger.info(f"开始分析股票: {selected_stocks}")
        
        analyzer = StockAnalyzer(selected_stocks)
        report_path = analyzer.run_analysis()
        
        relative_path = os.path.relpath(report_path, start=os.getcwd())
        
        app.logger.info(f"股票分析完成，报告路径: {relative_path}")
        return jsonify({
            "status": "success",
            "report_url": f"/{relative_path.replace(os.sep, '/')}"
        })
    
    except Exception as e:
        app.logger.error(f"股票分析失败: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/public/<path:filename>')
def serve_report(filename):
    app.logger.info(f"访问报告文件: {filename}")
    return send_file(f'public/{filename}')

if __name__ == '__main__':
    app.logger.info("应用启动")
    app.run(host='0.0.0.0', port=8000) 