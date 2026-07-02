"""
Agent 共享工具：LLM 客户端、Prompt 加载、配置读取
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger

# 显式指定 .env 文件路径，确保从项目根目录加载
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@lru_cache()
def load_settings() -> dict:
    """加载 settings.yaml"""
    path = CONFIG_DIR / "settings.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache()
def load_prompts() -> dict:
    """加载 prompts.yaml"""
    path = CONFIG_DIR / "prompts.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_llm(role: str = "planner", temperature: Optional[float] = None) -> ChatOpenAI:
    """根据角色获取 LLM 实例

    支持的 provider：
    - qwen: 通义千问（需 QWEN_API_KEY）
    - deepseek: DeepSeek（需 DEEPSEEK_API_KEY）
    - openai: OpenAI（需 OPENAI_API_KEY）
    - ollama: 本地 Ollama（需 OLLAMA_BASE_URL，默认 http://localhost:11434/v1）
    - siliconflow: 硅基流动（需 SILICONFLOW_API_KEY，OpenAI 兼容 API）

    Args:
        role: planner/executor/reviewer/interaction
        temperature: 温度参数，None 则用配置默认值

    Returns:
        ChatOpenAI 实例
    """
    settings = load_settings()
    llm_config = settings.get("llm", {})
    provider = os.getenv("LLM_PROVIDER", llm_config.get("provider", "qwen"))
    models = llm_config.get("models", {}).get(provider, {})
    model_name = models.get(role, llm_config.get("default_model", "qwen-max"))

    temp = temperature if temperature is not None else llm_config.get("temperature", 0.0)
    # M2 迭代修复：统一加 request_timeout，防止 LLM 调用无限卡住
    request_timeout = llm_config.get("request_timeout", 120)
    # M4-14: LLM 重试（读取 settings.yaml retry 配置，tenacity 已装但用 openai client 内置重试）
    retry_config = llm_config.get("retry", {})
    max_retries = retry_config.get("max_attempts", 3)

    # 根据 provider 构造不同的客户端
    if provider == "qwen":
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            openai_api_key=os.getenv("QWEN_API_KEY", ""),
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
    elif provider == "deepseek":
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            openai_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            openai_api_base="https://api.deepseek.com",
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
    elif provider == "ollama":
        # Ollama 通过 OpenAI 兼容 API 运行
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            openai_api_key="ollama",  # Ollama 不需要真实 key，但不能为空
            openai_api_base=base_url,
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
    elif provider == "siliconflow":
        # 硅基流动 OpenAI 兼容 API
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            openai_api_key=os.getenv("SILICONFLOW_API_KEY", ""),
            openai_api_base="https://api.siliconflow.cn/v1",
            request_timeout=request_timeout,
            max_retries=max_retries,
        )
    else:
        raise ValueError(f"不支持的 LLM provider: {provider}，可选: qwen/deepseek/openai/ollama/siliconflow")


def get_prompt(role: str) -> str:
    """获取指定角色的 prompt 模板"""
    prompts = load_prompts()
    if role not in prompts:
        raise KeyError(f"Prompt 角色不存在: {role}，可用: {list(prompts.keys())}")
    return prompts[role]


def get_process_constraints(process_type: str = "heat_treatment") -> dict:
    """获取工艺约束"""
    settings = load_settings()
    return settings.get("process_constraints", {}).get(process_type, {})


# ===== M4-9: 产线配置层 =====

@lru_cache()
def load_line_config(line_id: str = "heat_treatment") -> dict:
    """加载产线配置（M4-9 多产线抽象层）

    从 config/lines/<line_id>.yaml 加载产线专属配置（标准参数、约束、缺陷类型等）。
    若配置文件不存在，回退到 heat_treatment 配置。

    Args:
        line_id: 产线ID（如 heat_treatment / welding / rolling）

    Returns:
        产线配置 dict，包含 line_id/name/process_type/standard_params/constraints/defect_types 等
    """
    lines_dir = CONFIG_DIR / "lines"
    path = lines_dir / f"{line_id}.yaml"
    if not path.exists():
        logger.warning(f"产线配置不存在: {line_id}，回退到 heat_treatment")
        path = lines_dir / "heat_treatment.yaml"
        if not path.exists():
            return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_line_standard_params(line_id: str = "heat_treatment") -> dict:
    """获取产线标准工艺参数（M4-9）

    替代原硬编码的 840/120/5.0 等阈值。

    Args:
        line_id: 产线ID

    Returns:
        标准参数 dict，如 {"temperature": 840, "holding_time": 120, "cooling_rate": 5.0}
    """
    config = load_line_config(line_id)
    return config.get("standard_params", {})


def get_line_constraints(line_id: str = "heat_treatment") -> dict:
    """获取产线工艺约束（M4-9）

    Args:
        line_id: 产线ID

    Returns:
        约束 dict，如 {"temperature_min": 800, "temperature_max": 1100, ...}
    """
    config = load_line_config(line_id)
    return config.get("constraints", {})


def list_available_lines() -> list[str]:
    """列出全部可用产线ID（M4-9）

    扫描 config/lines/ 目录下的 yaml 文件。

    Returns:
        产线ID列表，如 ["heat_treatment", "welding"]
    """
    lines_dir = CONFIG_DIR / "lines"
    if not lines_dir.exists():
        return ["heat_treatment"]
    return [
        f.stem for f in lines_dir.glob("*.yaml")
    ]


# ===== M4-10: 多租户权限隔离 =====


@lru_cache()
def load_line_access() -> dict:
    """加载产线访问权限配置（M4-10 多租户）

    从 config/line_access.yaml 加载用户-产线映射、角色权限矩阵。

    Returns:
        权限配置 dict，结构：
        {
            "default_role": "operator",
            "roles": {role: {allowed_lines, can_write, can_delete}},
            "users": {user_id: {role, allowed_lines}}
        }
        配置文件不存在时返回空 dict（降级为不鉴权）。
    """
    path = CONFIG_DIR / "line_access.yaml"
    if not path.exists():
        logger.warning("[M4-10] line_access.yaml 不存在，权限隔离降级为全部放行")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_user_role(user_id: str) -> str:
    """获取用户角色（M4-10）

    Args:
        user_id: 用户ID

    Returns:
        角色名（admin/supervisor/operator），未知用户走 default_role
    """
    access = load_line_access()
    if not access:
        return "admin"  # 降级模式：无配置则视为 admin
    users = access.get("users", {})
    user_cfg = users.get(user_id, {})
    if "role" in user_cfg:
        return user_cfg["role"]
    return access.get("default_role", "operator")


def get_user_lines(user_id: str) -> list[str]:
    """获取用户可访问的产线ID列表（M4-10）

    优先级：
    1. 用户配置中的 allowed_lines（覆盖角色默认）
    2. 角色配置中的 allowed_lines
    3. admin 角色返回 ["*"]（全部产线）

    Args:
        user_id: 用户ID

    Returns:
        产线ID列表，如 ["heat_treatment", "welding"]，或 ["*"] 表示全部
    """
    access = load_line_access()
    if not access:
        return ["*"]  # 降级模式：全部放行

    users = access.get("users", {})
    user_cfg = users.get(user_id, {})
    role = user_cfg.get("role", access.get("default_role", "operator"))
    roles = access.get("roles", {})
    role_cfg = roles.get(role, {})

    # 用户显式配置优先
    allowed = user_cfg.get("allowed_lines", role_cfg.get("allowed_lines", []))
    return allowed if allowed else []


def can_access_line(user_id: str, line_id: str) -> bool:
    """校验用户是否有权访问指定产线（M4-10）

    Args:
        user_id: 用户ID
        line_id: 产线ID

    Returns:
        是否允许访问
    """
    allowed = get_user_lines(user_id)
    if "*" in allowed:
        return True
    return line_id in allowed


def get_user_permissions(user_id: str) -> dict:
    """获取用户权限标识（M4-10）

    Args:
        user_id: 用户ID

    Returns:
        {"role": str, "can_write": bool, "can_delete": bool, "allowed_lines": list}
    """
    access = load_line_access()
    if not access:
        return {
            "role": "admin",
            "can_write": True,
            "can_delete": True,
            "allowed_lines": ["*"],
        }

    role = get_user_role(user_id)
    roles = access.get("roles", {})
    role_cfg = roles.get(role, {})
    return {
        "role": role,
        "can_write": bool(role_cfg.get("can_write", False)),
        "can_delete": bool(role_cfg.get("can_delete", False)),
        "allowed_lines": get_user_lines(user_id),
    }


def setup_tracing():
    """配置 LangSmith trace

    通过环境变量启用 LangChain 全链路追踪。
    需要设置 .env 中的 LANGSMITH_API_KEY 和 LANGSMITH_PROJECT。
    API key 为占位符时不启用，避免 403 错误。
    """
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    project = os.getenv("LANGSMITH_PROJECT", "metacraft-agent")

    # 排除占位符
    if api_key and not api_key.startswith("your_"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        logger.info(f"LangSmith trace 已启用, project={project}")
    else:
        # 显式禁用，避免 langchain 自动尝试
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning("LANGSMITH_API_KEY 未设置或为占位符，trace 已禁用（不影响运行）")


def extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON

    处理以下情况：
    1. 纯 JSON 字符串
    2. ```json ... ``` 代码块包裹
    3. ``` ... ``` 代码块包裹
    4. 文本中嵌入的 JSON（找第一个 { 到最后一个 }）

    Args:
        text: LLM 输出文本

    Returns:
        解析后的 dict，失败返回 None
    """
    import json
    import re

    if not text:
        return None

    text = text.strip()

    # 1. 直接尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 尝试从 ```json ... ``` 代码块提取
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 尝试从 ``` ... ``` 代码块提取
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 4. 尝试找第一个 { 到最后一个 }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None
