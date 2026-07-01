"""
API 鉴权模块
对应 TDD 第 10 节

JWT + RBAC：
- 操作员（operator）：可查询、提交反馈
- 工艺工程师（engineer）：可审核参数调整
- 管理员（admin）：全部权限
"""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel

# TODO: 实际项目中从环境变量读取
JWT_SECRET = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 8

security = HTTPBearer()


class User(BaseModel):
    """用户模型"""
    user_id: str
    username: str
    role: str  # operator / engineer / admin
    exp: Optional[datetime] = None


class Permission:
    """权限常量"""
    ANALYZE = "analyze"           # 提交归因
    VIEW_RESULT = "view_result"   # 查看结果
    SUBMIT_FEEDBACK = "feedback"  # 提交反馈
    APPROVE_ADJUSTMENT = "approve"  # 审核参数调整
    MANAGE_CASES = "manage_cases"  # 管理案例库
    VIEW_DASHBOARD = "dashboard"   # 查看看板


# 角色 → 权限映射
ROLE_PERMISSIONS = {
    "operator": [
        Permission.ANALYZE,
        Permission.VIEW_RESULT,
        Permission.SUBMIT_FEEDBACK,
    ],
    "engineer": [
        Permission.ANALYZE,
        Permission.VIEW_RESULT,
        Permission.SUBMIT_FEEDBACK,
        Permission.APPROVE_ADJUSTMENT,
        Permission.MANAGE_CASES,
    ],
    "admin": [
        Permission.ANALYZE,
        Permission.VIEW_RESULT,
        Permission.SUBMIT_FEEDBACK,
        Permission.APPROVE_ADJUSTMENT,
        Permission.MANAGE_CASES,
        Permission.VIEW_DASHBOARD,
    ],
}


def create_token(user: User) -> str:
    """生成 JWT token

    TODO: 使用 jose 或 pyjwt 实现
    """
    import jwt
    payload = {
        "user_id": user.user_id,
        "username": user.username,
        "role": user.role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> User:
    """验证 JWT token"""
    import jwt
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return User(
            user_id=payload["user_id"],
            username=payload["username"],
            role=payload["role"],
            exp=datetime.fromtimestamp(payload["exp"]),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Token",
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> User:
    """获取当前用户"""
    return verify_token(credentials.credentials)


def require_permission(permission: str):
    """权限校验依赖

    用法：
        @app.post("/api/v1/analyze", dependencies=[Depends(require_permission(Permission.ANALYZE))])
    """
    def permission_checker(user: User = Depends(get_current_user)) -> User:
        allowed = ROLE_PERMISSIONS.get(user.role, [])
        if permission not in allowed:
            logger.warning(f"权限拒绝: user={user.username}, role={user.role}, need={permission}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"权限不足，需要 {permission} 权限",
            )
        return user
    return permission_checker
