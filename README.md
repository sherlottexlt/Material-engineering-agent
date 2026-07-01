# MetaCraft Agent

面向材料加工产线的智能工艺优化 Agent。

## 快速开始

### 1. 环境准备
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置
```bash
cp .env.example .env
# 编辑 .env 填入 API key
```

### 3. 启动依赖服务
```bash
docker compose up -d chroma postgres langfuse
```

### 4. 初始化数据库
```bash
python scripts/init_db.py
```

### 5. 导入工艺手册
```bash
python scripts/ingest_handbooks.py data/handbooks/
```

### 6. 启动应用
```bash
# 启动 API 服务
uvicorn api.routes:app --reload --port 8000

# 启动 UI（新终端）
streamlit run ui/streamlit_app.py
```

## 项目结构
```
metacraft-agent/
├── agent/           # Agent 核心（编排、节点、Prompt、记忆）
├── mcp_servers/     # MCP 工具服务器
├── models/          # 数据模型
├── api/             # REST API
├── ui/              # Streamlit 界面
├── data/            # 数据（手册、案例、向量库）
├── eval/            # 评估脚本
├── tests/           # 测试
├── config/          # 配置文件
├── scripts/         # 工具脚本
└── docs/            # 文档（PRD/TDD/IMP）
```

## 文档
- [PRD.md](./PRD.md) - 产品需求文档
- [TDD.md](./TDD.md) - 技术设计文档
- [IMP.md](./IMP.md) - 实施计划

## 技术栈
- LangGraph（Agent 编排）
- MCP（工具协议）
- Chroma（向量记忆）
- FastAPI + Streamlit（接口）
- LangSmith + Langfuse（可观测）
