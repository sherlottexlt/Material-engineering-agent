"""
M2-11 流程配置化测试

测试 config/flows.yaml 加载和不同流程图的构建。
"""
import pytest

from agent.flow_config import (
    FlowConfig,
    load_flow_config,
    list_flows,
    clear_cache,
)


# ===== flow_config 测试 =====

class TestFlowConfigLoad:
    """测试流程配置加载"""

    def setup_method(self):
        """每个测试前清除缓存"""
        clear_cache()

    def test_load_parallel(self):
        """加载 parallel 流程"""
        config = load_flow_config("parallel")
        assert config.name == "parallel"
        assert config.mode == "parallel"
        assert "data" in config.parallel_agents
        assert "mechanism" in config.parallel_agents
        assert "knowledge" in config.parallel_agents
        assert config.enable_arbitrate is True

    def test_load_sequential(self):
        """加载 sequential 流程"""
        config = load_flow_config("sequential")
        assert config.name == "sequential"
        assert config.mode == "sequential"
        assert config.parallel_agents == []
        assert config.enable_arbitrate is False

    def test_load_data_first(self):
        """加载 data_first 流程"""
        config = load_flow_config("data_first")
        assert config.name == "data_first"
        assert config.mode == "hybrid"
        assert "data" in config.sequential_before
        assert "mechanism" in config.parallel_agents
        assert "knowledge" in config.parallel_agents

    def test_load_quick(self):
        """加载 quick 流程（跳过 mechanism/knowledge）"""
        config = load_flow_config("quick")
        assert config.name == "quick"
        assert config.mode == "sequential"
        assert "mechanism" in config.skip_agents
        assert "knowledge" in config.skip_agents
        assert config.enable_arbitrate is False

    def test_load_knowledge_heavy(self):
        """加载 knowledge_heavy 流程"""
        config = load_flow_config("knowledge_heavy")
        assert config.name == "knowledge_heavy"
        assert config.mode == "hybrid"
        assert "knowledge" in config.sequential_before
        assert "data" in config.parallel_agents
        assert "mechanism" in config.parallel_agents

    def test_default_flow(self):
        """None 时加载默认流程"""
        config = load_flow_config(None)
        # 默认应该是 parallel
        assert config.name == "parallel"

    def test_unknown_flow_falls_back(self):
        """未知流程名回退到默认"""
        config = load_flow_config("nonexistent_flow")
        assert config.name == "parallel"

    def test_list_flows(self):
        """列出所有流程"""
        flows = list_flows()
        assert "parallel" in flows
        assert "sequential" in flows
        assert "data_first" in flows
        assert "quick" in flows
        assert "knowledge_heavy" in flows
        assert len(flows) >= 5


class TestFlowConfigProperties:
    """测试 FlowConfig 属性"""

    def setup_method(self):
        clear_cache()

    def test_is_parallel_true_for_parallel(self):
        """parallel 模式 is_parallel=True"""
        config = load_flow_config("parallel")
        assert config.is_parallel is True

    def test_is_parallel_false_for_sequential(self):
        """sequential 模式 is_parallel=False"""
        config = load_flow_config("sequential")
        assert config.is_parallel is False

    def test_is_parallel_true_for_hybrid(self):
        """hybrid 模式（有并行 Agent）is_parallel=True"""
        config = load_flow_config("data_first")
        assert config.is_parallel is True

    def test_all_agents_parallel(self):
        """parallel 流程包含所有 Agent"""
        config = load_flow_config("parallel")
        agents = config.all_agents
        assert "data" in agents
        assert "mechanism" in agents
        assert "knowledge" in agents
        assert "decision" in agents
        assert "review" in agents
        assert "interaction" in agents

    def test_all_agents_quick_excludes_skipped(self):
        """quick 流程不包含被跳过的 Agent"""
        config = load_flow_config("quick")
        agents = config.all_agents
        assert "data" in agents
        assert "mechanism" not in agents
        assert "knowledge" not in agents
        assert "decision" in agents

    def test_all_agents_sequential_no_arbitrate(self):
        """sequential 流程不包含 arbitrate（enable_arbitrate=False）"""
        config = load_flow_config("sequential")
        agents = config.all_agents
        assert "arbitrate" not in agents

    def test_frozen_config(self):
        """FlowConfig 是不可变的"""
        config = load_flow_config("parallel")
        with pytest.raises((AttributeError, TypeError)):
            config.mode = "sequential"

    def test_cache_returns_same_instance(self):
        """缓存返回同一实例"""
        clear_cache()
        c1 = load_flow_config("parallel")
        c2 = load_flow_config("parallel")
        assert c1 is c2

    def test_clear_cache_returns_new_instance(self):
        """清除缓存后返回新实例"""
        clear_cache()
        c1 = load_flow_config("parallel")
        clear_cache()
        c2 = load_flow_config("parallel")
        assert c1 is not c2


# ===== coordinator 图构建测试 =====

class TestFlowGraphBuild:
    """测试不同流程的图构建"""

    def setup_method(self):
        clear_cache()

    def test_parallel_graph_builds(self):
        """parallel 流程图构建成功"""
        from agent.nodes.coordinator import build_collaboration_graph
        graph = build_collaboration_graph("parallel")
        assert graph is not None
        nodes = set(graph.nodes.keys())
        assert "planner" in nodes
        assert "data" in nodes
        assert "mechanism" in nodes
        assert "knowledge" in nodes
        assert "arbitrate" in nodes
        assert "decision" in nodes
        assert "review" in nodes
        assert "interaction" in nodes
        assert "memory_writer" in nodes

    def test_sequential_graph_builds(self):
        """sequential 流程图构建成功，无 arbitrate"""
        from agent.nodes.coordinator import build_collaboration_graph
        graph = build_collaboration_graph("sequential")
        nodes = set(graph.nodes.keys())
        assert "planner" in nodes
        assert "data" in nodes
        assert "mechanism" in nodes
        assert "knowledge" in nodes
        assert "arbitrate" not in nodes  # sequential 不启用仲裁
        assert "decision" in nodes
        assert "review" in nodes

    def test_quick_graph_skips_mechanism_knowledge(self):
        """quick 流程跳过 mechanism 和 knowledge"""
        from agent.nodes.coordinator import build_collaboration_graph
        graph = build_collaboration_graph("quick")
        nodes = set(graph.nodes.keys())
        assert "planner" in nodes
        assert "data" in nodes
        assert "mechanism" not in nodes  # 被跳过
        assert "knowledge" not in nodes  # 被跳过
        assert "arbitrate" not in nodes  # quick 不启用仲裁
        assert "decision" in nodes
        assert "review" in nodes
        assert "interaction" in nodes

    def test_data_first_graph_builds(self):
        """data_first 流程图构建成功"""
        from agent.nodes.coordinator import build_collaboration_graph
        graph = build_collaboration_graph("data_first")
        nodes = set(graph.nodes.keys())
        assert "planner" in nodes
        assert "data" in nodes
        assert "mechanism" in nodes
        assert "knowledge" in nodes
        assert "arbitrate" in nodes  # data_first 启用仲裁
        assert "decision" in nodes

    def test_knowledge_heavy_graph_builds(self):
        """knowledge_heavy 流程图构建成功"""
        from agent.nodes.coordinator import build_collaboration_graph
        graph = build_collaboration_graph("knowledge_heavy")
        nodes = set(graph.nodes.keys())
        assert "planner" in nodes
        assert "knowledge" in nodes
        assert "data" in nodes
        assert "mechanism" in nodes
        assert "arbitrate" in nodes

    def test_default_flow_builds(self):
        """None 时使用默认流程构建"""
        from agent.nodes.coordinator import build_collaboration_graph
        graph = build_collaboration_graph(None)
        nodes = set(graph.nodes.keys())
        assert "arbitrate" in nodes  # 默认 parallel 有 arbitrate

    def test_quick_has_fewer_nodes_than_parallel(self):
        """quick 流程节点数少于 parallel"""
        from agent.nodes.coordinator import build_collaboration_graph
        parallel_graph = build_collaboration_graph("parallel")
        quick_graph = build_collaboration_graph("quick")
        parallel_nodes = set(parallel_graph.nodes.keys())
        quick_nodes = set(quick_graph.nodes.keys())
        assert len(quick_nodes) < len(parallel_nodes)

    def test_build_orchestrator_accepts_flow_name(self):
        """build_orchestrator 接受 flow_name 参数"""
        from agent.orchestrator import build_orchestrator
        graph = build_orchestrator("quick")
        nodes = set(graph.nodes.keys())
        assert "mechanism" not in nodes  # quick 跳过 mechanism
