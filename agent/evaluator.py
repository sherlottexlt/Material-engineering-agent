"""
MetaCraft Agent 结果评估器
对应 TDD 第 4.7 节

三层评估：
1. LLM 自评（审核 Agent + 独立 LLM 双评）
2. 用户反馈采集
3. 效果跟踪（T+7 天）
"""
import json
from datetime import datetime, timedelta
from typing import Optional

from langchain_openai import ChatOpenAI
from loguru import logger

from models.entities import UserFeedback
from models.state import AgentState

EVALUATOR_PROMPT = """你是独立的评估 Agent，审核以下归因结果：
归因结论：{conclusion}
证据链：{evidence}

请评估：
1. 证据是否充分（1-5 分）
2. 推理是否合理（1-5 分）
3. 是否违反工艺常识（是/否）
4. 总体可信度（0-1）

输出 JSON：
{{
  "evidence_score": 1-5,
  "reasoning_score": 1-5,
  "violates_domain": true/false,
  "overall_confidence": 0.0-1.0,
  "comments": "评语"
}}
"""


class Evaluator:
    """结果评估器"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(model="qwen-max", temperature=0)
        # TODO: 初始化 LangSmith Client
        # self.langsmith = Client()

    async def evaluate(self, state: AgentState) -> dict:
        """对 Agent 输出进行 LLM 自评

        Args:
            state: Agent 最终状态

        Returns:
            评估结果字典
        """
        prompt = EVALUATOR_PROMPT.format(
            conclusion=state.get("final_answer", ""),
            evidence=json.dumps(state.get("observations", []), ensure_ascii=False, default=str),
        )

        try:
            resp = self.llm.invoke(prompt)
            score = json.loads(resp.content)
            logger.info(f"评估完成: confidence={score.get('overall_confidence')}")
            return score
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {"overall_confidence": 0.0, "comments": f"评估异常: {e}"}

    def collect_feedback(self, feedback: UserFeedback) -> bool:
        """收集用户反馈，更新记忆权重

        Args:
            feedback: 用户反馈对象

        Returns:
            是否成功
        """
        # TODO: 写入反馈表
        # TODO: 根据 proposal_id 找到对应 case，更新 confidence
        logger.info(f"收到反馈: proposal={feedback.proposal_id}, score={feedback.score}")
        return True

    async def track_effect(self, proposal_id: str, days: int = 7) -> Optional[dict]:
        """跟踪参数调整后的效果（T+N 天）

        Args:
            proposal_id: 建议ID
            days: 跟踪天数

        Returns:
            效果对比数据
        """
        # TODO: 查询调参后批次质量
        # TODO: 对比调整前后指标
        logger.info(f"跟踪建议 {proposal_id} 的 {days} 天效果")
        return None

    def export_eval_dataset(self, output_path: str):
        """导出评估数据集到 LangSmith

        Args:
            output_path: 输出路径
        """
        # TODO: 从历史执行记录导出
        pass
