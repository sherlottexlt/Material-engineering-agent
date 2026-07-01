"""
Knowledge MCP Server
对应 TDD 第 4.3 节

提供工具：
- search_handbook: 检索工艺手册
- search_cases: 检索历史案例
- write_case: 写入新案例
- update_case_confidence: 更新案例置信度

启动方式：python mcp_servers/knowledge_server.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# 添加项目根目录到 path，便于导入
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.memory.memory_service import MemoryService  # noqa: E402

server = Server("knowledge-tools")

# 全局 MemoryService 实例（延迟初始化）
_memory: Optional[MemoryService] = None


def get_memory() -> MemoryService:
    global _memory
    if _memory is None:
        _memory = MemoryService()
    return _memory


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_handbook",
            description="检索工艺手册内容（热处理规范、材料性能等）",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词"},
                    "top_k": {"type": "integer", "default": 3, "description": "返回条数"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_cases",
            description="检索历史缺陷归因案例",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词（缺陷类型/根因）"},
                    "top_k": {"type": "integer", "default": 3, "description": "返回条数"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="write_case",
            description="写入新的历史案例到长期记忆",
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string", "description": "案例ID"},
                    "defect_type": {"type": "string"},
                    "root_cause": {"type": "string"},
                    "solution": {"type": "string"},
                    "confidence": {"type": "number", "default": 0.5},
                    "batch_params": {"type": "object", "description": "批次工艺参数"},
                },
                "required": ["case_id", "defect_type", "root_cause", "solution"],
            },
        ),
        Tool(
            name="update_case_confidence",
            description="根据用户反馈更新案例置信度",
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "feedback_score": {"type": "number", "description": "反馈评分 0-1"},
                },
                "required": ["case_id", "feedback_score"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"知识工具调用: {name} | 参数: {arguments}")

    try:
        memory = get_memory()

        if name == "search_handbook":
            # TODO: 工艺手册单独建集合，当前复用 cases 集合
            results = memory.search_semantic(
                query=arguments["query"],
                top_k=arguments.get("top_k", 3),
            )
            result = {"source": "handbook", "results": results, "count": len(results)}

        elif name == "search_cases":
            results = memory.search_semantic(
                query=arguments["query"],
                top_k=arguments.get("top_k", 3),
            )
            result = {"source": "cases", "results": results, "count": len(results)}

        elif name == "write_case":
            from datetime import datetime
            from models.entities import BatchParams, CaseRecord

            batch_params = BatchParams(
                batch_id=arguments.get("batch_params", {}).get("batch_id", "unknown"),
                process_type=arguments.get("batch_params", {}).get("process_type", "heat_treatment"),
                temperature=arguments.get("batch_params", {}).get("temperature"),
                holding_time=arguments.get("batch_params", {}).get("holding_time"),
                cooling_rate=arguments.get("batch_params", {}).get("cooling_rate"),
                start_time=datetime.now(),
            )
            case = CaseRecord(
                case_id=arguments["case_id"],
                defect_type=arguments["defect_type"],
                batch_params=batch_params,
                root_cause=arguments["root_cause"],
                solution=arguments["solution"],
                confidence=arguments.get("confidence", 0.5),
                created_at=datetime.now(),
                source="auto",
            )
            success = memory.write_semantic(case)
            result = {"success": success, "case_id": case.case_id}

        elif name == "update_case_confidence":
            success = memory.update_confidence(
                case_id=arguments["case_id"],
                feedback_score=arguments["feedback_score"],
            )
            result = {"success": success, "case_id": arguments["case_id"]}

        else:
            result = {"error": f"未知工具: {name}"}

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
    except Exception as e:
        logger.error(f"知识工具调用失败 {name}: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    logger.info("启动 Knowledge MCP Server")
    asyncio.run(main())
