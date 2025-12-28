import os
import subprocess
import sys

# 设置环境变量以跳过电子邮件提示
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'false'

# 获取当前目录下的app.py路径
app_path = os.path.join(os.path.dirname(__file__), 'app.py')

# 构建Streamlit命令
command = [
    sys.executable,  # 使用当前Python解释器
    '-m', 'streamlit', 'run',
    app_path
]

print('正在启动乳腺癌风险预测系统...')
print(f'执行命令: {" ".join(command)}')

# 执行命令
subprocess.run(command, check=True)