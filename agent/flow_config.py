"""
协作流程配置加载（M2-11）

从 config/flows.yaml 加载流程定义，支持不同场景切换协作模式。

用法：
    from agent.flow_config import load_flow_config, FlowConfig

    config = load_flow_config("parallel")  # 或 "sequential" / "data_first" / "quick"
    graph = build_collaboration_graph(config)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class FlowConfig:
    """协作流程配置"""
    name: str                                   # 流程名
    description: str                            # 描述
    mode: str                                   # parallel / sequential / hybrid
    parallel_agents: list[str] = field(default_factory=list)      # 并行执行的 Agent
    sequential_before: list[str] = field(default_factory=list)    # 并行前串行执行
    sequential_after: list[str] = field(default_factory=list)     # 并行后串行执行
    skip_agents: list[str] = field(default_factory=list)          # 跳过的 Agent
    enable_arbitrate: bool = True               # 是否启用冲突仲裁

    @property
    def is_parallel(self) -> bool:
        """是否包含并行阶段"""
        return self.mode == "parallel" or (self.mode == "hybrid" and len(self.parallel_agents) > 1)

    @property
    def all_agents(self) -> list[str]:
        """流程中所有 Agent（按执行顺序）"""
        agents = list(self.sequential_before)
        agents.extend(self.parallel_agents)
        agents.extend(a for a in self.sequential_after if a != "arbitrate" or self.enable_arbitrate)
        # 移除跳过的
        agents = [a for a in agents if a not in self.skip_agents]
        return agents


# ===== 配置缓存 =====
_cache: dict[str, FlowConfig] = {}


def _load_yaml() -> dict:
    """加载 flows.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "flows.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_flow_config(name: Optional[str] = None) -> FlowConfig:
    """加载流程配置

    Args:
        name: 流程名。None 时使用 default_flow。

    Returns:
        FlowConfig 实例
    """
    if name in _cache:
        return _cache[name]

    data = _load_yaml()
    if not data:
        # YAML 不存在，返回默认并行配置
        config = FlowConfig(
            name="parallel",
            description="默认并行协作（YAML 未加载）",
            mode="parallel",
            parallel_agents=["data", "mechanism", "knowledge"],
            sequential_after=["arbitrate", "decision", "review", "interaction"],
            enable_arbitrate=True,
        )
        _cache[name or "parallel"] = config
        return config

    flows = data.get("flows", {})
    default = data.get("default_flow", "parallel")
    flow_name = name or default

    if flow_name not in flows:
        # 未知流程名，回退到默认
        flow_name = default

    flow_data = flows.get(flow_name, flows.get(default, {}))

    enable_arbitrate = flow_data.get("enable_arbitrate", True)
    sequential_after = list(flow_data.get("sequential_after", []))

    # 如果启用仲裁但 sequential_after 未包含 arbitrate，自动插入到开头
    if enable_arbitrate and "arbitrate" not in sequential_after and "arbitrate" not in flow_data.get("skip_agents", []):
        sequential_after = ["arbitrate"] + sequential_after

    config = FlowConfig(
        name=flow_name,
        description=flow_data.get("description", ""),
        mode=flow_data.get("mode", "parallel"),
        parallel_agents=list(flow_data.get("parallel_agents", [])),
        sequential_before=list(flow_data.get("sequential_before", [])),
        sequential_after=sequential_after,
        skip_agents=list(flow_data.get("skip_agents", [])),
        enable_arbitrate=enable_arbitrate,
    )

    _cache[flow_name] = config
    return config


def list_flows() -> list[str]:
    """列出所有可用流程名"""
    data = _load_yaml()
    return list(data.get("flows", {}).keys()) or ["parallel"]


def clear_cache():
    """清除配置缓存（测试用）"""
    _cache.clear()
