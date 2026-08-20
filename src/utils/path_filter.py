"""
文件路径过滤工具

支持通过 EXCLUDED_PATHS 环境变量配置需要排除的路径模式（glob 语法，逗号分隔）。

示例配置：
    EXCLUDED_PATHS=vendor/**,**/node_modules/**,**/*.min.js,**/dist/**,**/__pycache__/**

规则：
- 使用标准 glob 语法，** 匹配任意层级目录
- 路径分隔符统一用 /（Windows 反斜杠会自动转换）
- 大小写不敏感匹配
"""

import fnmatch
import os
from functools import lru_cache

from src.utils.log import logger

# 内置默认排除列表，覆盖绝大多数项目的第三方依赖和自动生成目录
_DEFAULT_EXCLUDED_PATTERNS = [
    "vendor/**",
    "**/vendor/**",
    "**/node_modules/**",
    "**/dist/**",
    "**/build/**",
    "**/.git/**",
    "**/__pycache__/**",
    "**/*.min.js",
    "**/*.min.css",
    "**/migrations/**",       # Django/Flask DB 迁移文件
    "**/generated/**",
    "**/auto_generated/**",
    "**/*.pb.go",             # protobuf 生成的 Go 文件
    "**/*_pb2.py",            # protobuf 生成的 Python 文件
    "**/*.g.dart",            # Flutter 代码生成文件
]


@lru_cache(maxsize=1)
def _get_excluded_patterns() -> list:
    """
    从环境变量读取排除模式列表，与内置默认列表合并。
    结果缓存在进程内，配置不会频繁变化。
    """
    env_val = os.getenv("EXCLUDED_PATHS", "")
    user_patterns = [p.strip() for p in env_val.split(",") if p.strip()] if env_val else []
    patterns = _DEFAULT_EXCLUDED_PATTERNS + user_patterns
    if user_patterns:
        logger.info(f"路径过滤：已加载用户自定义排除模式 {user_patterns}")
    return patterns


def is_path_excluded(file_path: str) -> bool:
    """
    判断文件路径是否应被排除。

    :param file_path: 文件相对路径，如 "vendor/lib/foo.py" 或 "src/main.go"
    :return: True 表示应排除，False 表示保留
    """
    if not file_path:
        return False

    # 统一路径分隔符为 /
    normalized = file_path.replace("\\", "/")

    for pattern in _get_excluded_patterns():
        # fnmatch 不原生支持 **，手动处理：
        # 将 ** 替换为一个能匹配任意字符（包括 /）的占位符，
        # 再用 fnmatch 做最终匹配
        if fnmatch.fnmatch(normalized, pattern):
            return True
        # 处理 **/ 前缀：pattern 如 **/node_modules/** 对 node_modules/foo.js 也应匹配
        if pattern.startswith("**/"):
            suffix_pattern = pattern[3:]  # 去掉 **/
            # 匹配路径中任意位置的子路径
            parts = normalized.split("/")
            for i in range(len(parts)):
                sub_path = "/".join(parts[i:])
                if fnmatch.fnmatch(sub_path, suffix_pattern):
                    return True

    return False


def filter_by_path(changes: list) -> list:
    """
    从 changes 列表中过滤掉路径被排除的文件。
    兼容 GitLab / GitHub / Gitea 三种格式（均含 new_path 字段）。

    :param changes: 原始 changes 列表
    :return: 过滤后的 changes 列表
    """
    result = []
    for item in changes:
        path = item.get("new_path") or item.get("filename") or item.get("old_path", "")
        if is_path_excluded(path):
            logger.info(f"路径过滤：跳过 {path}")
        else:
            result.append(item)

    excluded_count = len(changes) - len(result)
    if excluded_count:
        logger.info(f"路径过滤：共排除 {excluded_count} 个文件，保留 {len(result)} 个文件")

    return result
