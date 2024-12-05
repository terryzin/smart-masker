@echo off
echo Setting up Smart Masker development environment with Python 3.11...

REM 删除现有虚拟环境（如果存在）
if exist venv (
    call venv\Scripts\deactivate.bat
    rmdir /s /q venv
)

REM 使用 Python 3.11 创建新的虚拟环境
"C:\Users\chenlei\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv
call venv\Scripts\activate.bat

REM 更新基础工具
python -m pip install --upgrade pip setuptools wheel

REM 安装项目依赖
pip install -r requirements.txt

REM 创建前端目录（如果不存在）
if not exist frontend mkdir frontend

REM 初始化 React 应用
cd frontend
npx create-react-app . --template typescript
npm install @mui/material @emotion/react @emotion/styled @mui/icons-material axios

echo Setup complete! To start development:
echo 1. In one terminal:
echo    cd backend
echo    python -m uvicorn app.main:app --reload
echo 2. In another terminal:
echo    cd frontend
echo    npm start 