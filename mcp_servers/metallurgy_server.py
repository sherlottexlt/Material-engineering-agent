"""
Metallurgy MCP Server
对应 TDD 第 4.3.2 节

提供工具：
- run_metallurgy_model: 调用材料机理模型（JMAK 方程等）
- validate_hypothesis: 验证工艺假设

启动方式：python mcp_servers/metallurgy_server.py
"""
import asyncio
import json
import math

from loguru import logger
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("metallurgy-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_metallurgy_model",
            description="调用材料机理模型预测性能指标（基于 JMAK 方程、Hall-Petch 关系等）",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_type": {
                        "type": "string",
                        "enum": ["jmak", "hall_petch", "cooling_rate"],
                        "description": "模型类型",
                    },
                    "temperature": {"type": "number", "description": "温度 (℃)"},
                    "holding_time": {"type": "number", "description": "保温时间 (分钟)"},
                    "cooling_rate": {"type": "number", "description": "冷却速率 (℃/s)"},
                    "grain_size": {"type": "number", "description": "晶粒尺寸 (μm)，可选"},
                },
                "required": ["model_type"],
            },
        ),
        Tool(
            name="validate_hypothesis",
            description="验证工艺假设是否成立（如：保温时间不足是否导致硬度偏低）",
            inputSchema={
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string", "description": "假设描述"},
                    "actual_params": {"type": "object", "description": "实际工艺参数"},
                    "measured_value": {"type": "number", "description": "实测性能值"},
                    "standard_value": {"type": "number", "description": "标准性能值"},
                },
                "required": ["hypothesis", "actual_params"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"机理工具调用: {name} | 参数: {arguments}")

    try:
        if name == "run_metallurgy_model":
            result = run_metallurgy_model(
                model_type=arguments["model_type"],
                temperature=arguments.get("temperature"),
                holding_time=arguments.get("holding_time"),
                cooling_rate=arguments.get("cooling_rate"),
                grain_size=arguments.get("grain_size"),
            )
        elif name == "validate_hypothesis":
            result = validate_hypothesis(
                hypothesis=arguments["hypothesis"],
                actual_params=arguments["actual_params"],
                measured_value=arguments.get("measured_value"),
                standard_value=arguments.get("standard_value"),
            )
        else:
            result = {"error": f"未知工具: {name}"}

        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except Exception as e:
        logger.error(f"机理工具调用失败 {name}: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


def jmak_model(temperature: float, holding_time: float) -> dict:
    """JMAK 方程：预测相变分数

    f = 1 - exp(-k * t^n)

    TODO: 根据具体材料标定 k 和 n 参数
    """
    k = 0.01   # 速率常数（需标定）
    n = 1.5    # Avrami 指数（需标定）
    # 温度修正（简化）
    temp_factor = max(0.1, (temperature - 700) / 200) if temperature else 1.0

    fraction = 1 - math.exp(-k * (holding_time ** n) * temp_factor)
    predicted_hardness = 40 + 20 * fraction

    return {
        "model": "JMAK",
        "inputs": {"temperature": temperature, "holding_time": holding_time},
        "outputs": {
            "transformation_fraction": round(fraction, 4),
            "predicted_hardness_HRc": round(predicted_hardness, 2),
        },
        "parameters": {"k": k, "n": n, "temp_factor": round(temp_factor, 4)},
        "note": "简化模型，参数需根据实际材料标定",
    }


def hall_petch_model(grain_size: float) -> dict:
    """Hall-Petch 关系：晶粒尺寸与强度

    σ = σ0 + k * d^(-1/2)

    TODO: 标定 σ0 和 k
    """
    sigma_0 = 200  # 摩擦应力 (MPa)
    k = 0.5        # Hall-Petch 系数

    strength = sigma_0 + k * (grain_size ** -0.5)
    estimated_hardness = strength / 3  # 简化换算

    return {
        "model": "Hall-Petch",
        "inputs": {"grain_size_um": grain_size},
        "outputs": {
            "yield_strength_MPa": round(strength, 2),
            "estimated_hardness_HV": round(estimated_hardness, 2),
        },
        "parameters": {"sigma_0": sigma_0, "k": k},
    }


def cooling_rate_model(cooling_rate: float) -> dict:
    """冷却速率与硬度关系（简化）"""
    # 冷却速率越快，硬度越高（简化线性模型）
    base_hardness = 45
    hardness_increase = cooling_rate * 0.8

    return {
        "model": "cooling_rate",
        "inputs": {"cooling_rate": cooling_rate},
        "outputs": {
            "estimated_hardness_HRc": round(base_hardness + hardness_increase, 2),
        },
        "note": "简化线性模型，实际关系更复杂",
    }


def run_metallurgy_model(
    model_type: str,
    temperature: float | None,
    holding_time: float | None,
    cooling_rate: float | None,
    grain_size: float | None,
) -> dict:
    """运行机理模型"""
    if model_type == "jmak":
        if temperature is None or holding_time is None:
            return {"error": "JMAK 模型需要 temperature 和 holding_time"}
        return jmak_model(temperature, holding_time)
    elif model_type == "hall_petch":
        if grain_size is None:
            return {"error": "Hall-Petch 模型需要 grain_size"}
        return hall_petch_model(grain_size)
    elif model_type == "cooling_rate":
        if cooling_rate is None:
            return {"error": "冷却速率模型需要 cooling_rate"}
        return cooling_rate_model(cooling_rate)
    else:
        return {"error": f"未知模型类型: {model_type}"}


def validate_hypothesis(
    hypothesis: str,
    actual_params: dict,
    measured_value: float | None,
    standard_value: float | None,
) -> dict:
    """验证工艺假设

    TODO: 接入更复杂的验证逻辑
    当前仅做简单参数对比
    """
    result = {
        "hypothesis": hypothesis,
        "actual_params": actual_params,
        "verdict": "inconclusive",
        "evidence": [],
        "confidence": 0.5,
    }

    # 简单规则验证
    if measured_value and standard_value and measured_value < standard_value:
        result["evidence"].append(f"实测值 {measured_value} 低于标准值 {standard_value}")

        # 检查保温时间
        holding = actual_params.get("holding_time")
        if holding and holding < 60:
            result["evidence"].append(f"保温时间 {holding} 分钟偏短（建议 ≥ 60 分钟）")
            result["verdict"] = "supported"
            result["confidence"] = 0.7

        # 检查冷却速率
        cooling = actual_params.get("cooling_rate")
        if cooling and cooling < 1.0:
            result["evidence"].append(f"冷却速率 {cooling} ℃/s 偏低")
            result["verdict"] = "supported"
            result["confidence"] = max(result["confidence"], 0.75)

    return result


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    logger.info("启动 Metallurgy MCP Server")
    asyncio.run(main())
