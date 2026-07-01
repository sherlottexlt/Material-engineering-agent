"""角色 Prompt 与元数据"""
from agent.prompts.roles import (
    ROLES,
    PARALLEL_GROUP,
    SEQUENTIAL_AFTER_JOIN,
    Role,
    get_role,
    get_role_prompt,
    get_parallel_roles,
)

__all__ = [
    "ROLES",
    "PARALLEL_GROUP",
    "SEQUENTIAL_AFTER_JOIN",
    "Role",
    "get_role",
    "get_role_prompt",
    "get_parallel_roles",
]
