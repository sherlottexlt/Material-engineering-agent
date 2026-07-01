"""
MetaCraft Agent 核心实体定义
对应 TDD 第 3.1 节
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DefectType(str, Enum):
    """缺陷类型"""
    HARDNESS_LOW = "hardness_low"          # 硬度偏低
    HARDNESS_HIGH = "hardness_high"        # 硬度偏高
    DEFORMATION = "deformation"            # 变形
    CRACK = "crack"                        # 裂纹
    MICROSTRUCTURE = "microstructure"      # 组织异常
    OTHER = "other"


class ProcessType(str, Enum):
    """工艺类型"""
    HEAT_TREATMENT = "heat_treatment"      # 热处理
    WELDING = "welding"                    # 焊接
    ROLLING = "rolling"                    # 轧制
    FORGING = "forging"                    # 锻造


class BatchParams(BaseModel):
    """批次工艺参数"""
    batch_id: str = Field(description="批次编号")
    process_type: ProcessType = Field(description="工艺类型")
    temperature: Optional[float] = Field(default=None, description="温度 (℃)")
    holding_time: Optional[float] = Field(default=None, description="保温时间 (分钟)")
    cooling_rate: Optional[float] = Field(default=None, description="冷却速率 (℃/s)")
    pressure: Optional[float] = Field(default=None, description="压力 (MPa)")
    raw_material_batch: Optional[str] = Field(default=None, description="原材料批次")
    start_time: datetime = Field(description="开始时间")
    end_time: Optional[datetime] = Field(default=None, description="结束时间")


class DefectRecord(BaseModel):
    """缺陷记录"""
    record_id: str = Field(description="缺陷记录ID")
    batch_id: str = Field(description="批次编号")
    defect_type: DefectType = Field(description="缺陷类型")
    severity: float = Field(ge=0, le=1, description="严重程度 0-1")
    measured_value: Optional[float] = Field(default=None, description="实测值")
    standard_value: Optional[float] = Field(default=None, description="标准值")
    detected_at: datetime = Field(description="检测时间")
    description: Optional[str] = Field(default=None, description="缺陷描述")


class CaseRecord(BaseModel):
    """历史案例（长期记忆）"""
    case_id: str = Field(description="案例ID")
    defect_type: DefectType = Field(description="缺陷类型")
    batch_params: BatchParams = Field(description="批次参数")
    root_cause: str = Field(description="根因分析")
    solution: str = Field(description="解决方案")
    effect: Optional[str] = Field(default=None, description="调整后效果")
    confidence: float = Field(default=0.5, ge=0, le=1, description="置信度")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    source: str = Field(default="auto", description="来源: manual/auto")
    tags: list[str] = Field(default_factory=list, description="标签")


class AdjustmentProposal(BaseModel):
    """参数调整建议"""
    proposal_id: str = Field(description="建议ID")
    batch_id: str = Field(description="批次编号")
    adjustments: dict[str, float] = Field(description="参数调整项 {参数名: 调整量}")
    expected_effect: str = Field(description="预期效果")
    risks: list[str] = Field(default_factory=list, description="风险提示")
    evidence: list[str] = Field(default_factory=list, description="证据链")
    confidence: float = Field(default=0.5, ge=0, le=1, description="置信度")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    status: str = Field(default="pending", description="状态: pending/approved/rejected")


class UserFeedback(BaseModel):
    """用户反馈"""
    feedback_id: str = Field(description="反馈ID")
    proposal_id: str = Field(description="关联建议ID")
    user_id: str = Field(description="用户ID")
    action: str = Field(description="动作: adopted/rejected/partial")
    score: float = Field(ge=0, le=1, description="评分")
    comment: Optional[str] = Field(default=None, description="评论")
    created_at: datetime = Field(default_factory=datetime.now)


class ProcessConstraints(BaseModel):
    """工艺约束"""
    temperature_min: Optional[float] = None
    temperature_max: Optional[float] = None
    holding_time_min: Optional[float] = None
    holding_time_max: Optional[float] = None
    cooling_rate_min: Optional[float] = None
    cooling_rate_max: Optional[float] = None

    def validate_params(self, params: BatchParams) -> list[str]:
        """校验参数是否在约束范围内，返回违规项列表"""
        violations = []
        if self.temperature_min and params.temperature:
            if params.temperature < self.temperature_min:
                violations.append(f"温度 {params.temperature}℃ 低于下限 {self.temperature_min}℃")
        if self.temperature_max and params.temperature:
            if params.temperature > self.temperature_max:
                violations.append(f"温度 {params.temperature}℃ 高于上限 {self.temperature_max}℃")
        # TODO: 补充其他参数校验
        return violations
