"""
MES MCP Server
对应 TDD 第 4.3.1 节

提供工具：
- query_batch_params: 查询批次工艺参数
- query_defect_history: 查询历史缺陷记录
- submit_adjustment: 提交参数调整建议

启动方式：python mcp_servers/mes_server.py
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Any

import httpx
from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("mes-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """声明可用工具"""
    return [
        Tool(
            name="query_batch_params",
            description="根据批次ID查询工艺参数（温度、保温时间、冷却速率等）",
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "批次编号"},
                },
                "required": ["batch_id"],
            },
        ),
        Tool(
            name="query_defect_history",
            description="查询历史缺陷记录，可按缺陷类型和时间过滤",
            inputSchema={
                "type": "object",
                "properties": {
                    "defect_type": {
                        "type": "string",
                        "description": "缺陷类型: hardness_low/hardness_high/deformation/crack",
                    },
                    "days_back": {"type": "integer", "default": 30, "description": "查询天数"},
                    "limit": {"type": "integer", "default": 50, "description": "返回条数上限"},
                },
            },
        ),
        Tool(
            name="submit_adjustment",
            description="提交参数调整建议到审核流程（不会直接修改产线参数）",
            inputSchema={
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "批次编号"},
                    "adjustments": {
                        "type": "object",
                        "description": "参数调整项 {参数名: 调整量}",
                    },
                    "reason": {"type": "string", "description": "调整原因"},
                },
                "required": ["batch_id", "adjustments"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """处理工具调用"""
    logger.info(f"MES 工具调用: {name} | 参数: {arguments}")

    try:
        if name == "query_batch_params":
            result = await query_batch_params(arguments["batch_id"])
        elif name == "query_defect_history":
            result = await query_defect_history(
                defect_type=arguments.get("defect_type"),
                days_back=arguments.get("days_back", 30),
                limit=arguments.get("limit", 50),
            )
        elif name == "submit_adjustment":
            result = await submit_adjustment(
                batch_id=arguments["batch_id"],
                adjustments=arguments["adjustments"],
                reason=arguments.get("reason", ""),
            )
        else:
            result = {"error": f"未知工具: {name}"}

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
    except Exception as e:
        logger.error(f"工具调用失败 {name}: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


async def query_batch_params(batch_id: str) -> dict:
    """查询批次工艺参数

    TODO: 接入真实 MES 系统
    当前返回模拟数据
    """
    mes_api = os.getenv("MES_API_BASE", "")
    mes_key = os.getenv("MES_API_KEY", "")

    if mes_api:
        # 真实接入
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{mes_api}/batches/{batch_id}",
                headers={"Authorization": f"Bearer {mes_key}"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
    else:
        # 模拟数据
        return {
            "batch_id": batch_id,
            "process_type": "heat_treatment",
            "temperature": 850,
            "holding_time": 120,
            "cooling_rate": 5.0,
            "raw_material_batch": f"RM-{batch_id[-4:]}",
            "start_time": "2026-06-28T08:00:00",
            "end_time": "2026-06-28T10:00:00",
            "_note": "模拟数据，未接入真实 MES",
        }


async def query_defect_history(
    defect_type: str | None,
    days_back: int,
    limit: int,
) -> dict:
    """查询历史缺陷记录

    TODO: 接入真实质量系统
    """
    # 模拟数据
    return {
        "total": 2,
        "records": [
            {
                "record_id": "DR-20260620-001",
                "batch_id": "B20260620-A",
                "defect_type": "hardness_low",
                "measured_value": 52.3,
                "standard_value": 58.0,
                "root_cause": "保温时间不足",
                "created_at": "2026-06-20T14:00:00",
            },
            {
                "record_id": "DR-20260615-003",
                "batch_id": "B20260615-C",
                "defect_type": "hardness_low",
                "measured_value": 54.1,
                "standard_value": 58.0,
                "root_cause": "冷却速率过低",
                "created_at": "2026-06-15T16:00:00",
            },
        ],
        "_note": "模拟数据",
        "filter": {"defect_type": defect_type, "days_back": days_back, "limit": limit},
    }


async def submit_adjustment(batch_id: str, adjustments: dict, reason: str) -> dict:
    """提交参数调整建议

    注意：此操作只是提交建议，不直接修改产线参数
    需要人工审核通过后才会执行
    """
    proposal_id = f"PA-{int(datetime.now().timestamp())}"
    return {
        "proposal_id": proposal_id,
        "batch_id": batch_id,
        "adjustments": adjustments,
        "reason": reason,
        "status": "pending_review",
        "submitted_at": datetime.now().isoformat(),
        "message": "已提交审核，等待工艺工程师确认",
    }


async def main():
    """启动 MCP Server（stdio 模式）"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    logger.info("启动 MES MCP Server")
    asyncio.run(main())
