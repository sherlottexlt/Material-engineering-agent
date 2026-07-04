"""
MetaCraft Agent 结果评估器
对应 TDD 第 4.7 节

三层评估：
1. LLM 自评（审核 Agent + 独立 LLM 双评）
2. 用户反馈采集（M5-2: collect_feedback 写入反馈表 + 聚合更新 confidence）
3. 效果跟踪（M5-1/M5-2: T+7 天跟踪 + 归因到案例 confidence）
"""
import json
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

    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        memory=None,
        effect_tracker=None,
    ):
        self.llm = llm or ChatOpenAI(model="qwen-max", temperature=0)
        # M5-2: 注入 MemoryService 和 EffectTracker 以支持反馈归因
        self.memory = memory
        self.effect_tracker = effect_tracker
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

    def collect_feedback(self, feedback: UserFeedback) -> dict:
        """M5-2: 收集用户反馈，持久化并聚合更新案例 confidence

        流程：
        1. write_feedback 持久化到 SQLite feedback 表
        2. aggregate_feedback_update 聚合该案例所有反馈，时间衰减加权更新 confidence
           （多条反馈比单条更可信，反馈分占 50%，旧 confidence 占 50%）

        Args:
            feedback: 用户反馈对象

        Returns:
            {"saved": bool, "feedback_id": str, "confidence_update": dict}
        """
        if self.memory is None:
            logger.warning("Evaluator 未注入 memory，collect_feedback 跳过")
            return {"saved": False, "feedback_id": feedback.feedback_id,
                    "confidence_update": None, "error": "memory 未注入"}

        # 1. 持久化反馈
        line_id = getattr(feedback, "line_id", "heat_treatment")
        saved = self.memory.write_feedback(
            feedback_id=feedback.feedback_id,
            proposal_id=feedback.proposal_id,
            user_id=feedback.user_id,
            action=feedback.action,
            score=feedback.score,
            comment=feedback.comment,
            line_id=line_id,
        )
        if not saved:
            logger.error(f"反馈持久化失败: {feedback.feedback_id}")
            return {"saved": False, "feedback_id": feedback.feedback_id,
                    "confidence_update": None, "error": "write_feedback 失败"}

        # 2. 聚合更新 confidence（用 proposal_id 作为 case_id 查询关联反馈）
        update_result = self.memory.aggregate_feedback_update(
            case_id=feedback.proposal_id, days=90
        )

        logger.info(
            f"M5-2 反馈已收集: feedback={feedback.feedback_id}, "
            f"proposal={feedback.proposal_id}, action={feedback.action}, "
            f"confidence: {update_result.get('old_confidence')} → "
            f"{update_result.get('new_confidence')} ({update_result.get('feedback_count')} 条反馈)"
        )
        return {
            "saved": True,
            "feedback_id": feedback.feedback_id,
            "confidence_update": update_result,
        }

    async def track_effect(self, proposal_id: str, days: int = 7) -> Optional[dict]:
        """M5-1/M5-2: 跟踪参数调整后的效果（T+N 天）+ 归因到案例

        委托给 EffectTracker.track_effect，并在跟踪完成后调用
        attribute_effect 把效果反馈到案例 confidence。

        Args:
            proposal_id: 建议ID（需有对应的 pending 跟踪记录）
            days: 跟踪天数（仅用于日志，实际跟踪由 schedule_tracking 决定）

        Returns:
            效果对比 + 归因结果；无跟踪记录返回 None
        """
        if self.effect_tracker is None:
            logger.warning("Evaluator 未注入 effect_tracker，track_effect 跳过")
            return None

        # 通过 proposal_id 查找跟踪记录
        records = self.effect_tracker.list_trackings(days=365, limit=500)
        target = None
        for rec in records:
            if rec.get("proposal_id") == proposal_id:
                target = rec
                break

        if target is None:
            logger.warning(f"未找到 proposal={proposal_id} 的跟踪记录")
            return None

        tracking_id = target["tracking_id"]
        # 执行跟踪
        track_result = self.effect_tracker.track_effect(tracking_id)
        if track_result is None:
            return None

        # M5-2: 自动归因
        attr_result = self.effect_tracker.attribute_effect(tracking_id)

        logger.info(f"M5-1/2 跟踪+归因完成: proposal={proposal_id}, days={days}")
        return {
            "tracking": track_result,
            "attribution": attr_result,
        }

    def export_eval_dataset(self, output_path: str):
        """导出评估数据集到 LangSmith

        Args:
            output_path: 输出路径
        """
        # TODO: 从历史执行记录导出
        pass
