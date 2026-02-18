# 快速启动

## 环境要求

- Node.js >= 18.13, <= 22.x
- Python >= 3.11, < 3.13
- pnpm (推荐) 或 npm
- uv (Python 包管理)

## 安装依赖

```bash
# 前端
pnpm install

# 后端（使用 uv 创建虚拟环境 + 安装依赖）
uv venv --python 3.11
uv pip install -r backend/requirements.txt
```

## 启动开发服务

```bash
# 终端 1：启动后端
source .venv/bin/activate
bash backend/dev.sh
# 后端运行在 http://localhost:8080

# 终端 2：启动前端
pnpm dev
# 前端运行在 http://localhost:5173
```

首次访问会提示注册管理员账号。模型 API（OpenAI / Ollama 等）在前端 Settings > Connections 中配置。

## Docker 启动

```bash
docker-compose up -d
# 访问 http://localhost:3000
```
