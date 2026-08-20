"""
MR/PR diff 去重缓存

功能：对 changes 内容计算 MD5 hash，相同 hash 在 TTL 时间内视为重复审查，直接跳过 LLM 调用。

使用场景：
- 同一个 MR/PR 连续触发多次 webhook（如 fix typo、补充注释等微小改动后 force push）
- 同一批文件内容实际没有变化但 webhook 被重复发送

配置：
- DIFF_CACHE_TTL：缓存有效期（秒），默认 3600（1小时）
  超过 TTL 后同一 hash 的 MR 会被重新审查，适应"改了再提"的情况
- DIFF_DEDUP_ENABLED：是否启用去重，默认 1（启用），设为 0 可关闭
"""

import hashlib
import os
import threading
import time

from src.utils.log import logger

_cache_lock = threading.Lock()

# { diff_hash: expire_timestamp }
_hash_cache: dict[str, float] = {}


def _get_ttl() -> int:
    return int(os.getenv("DIFF_CACHE_TTL", 3600))


def _cleanup_expired():
    """清理过期条目，避免内存无限增长（在锁内调用）。"""
    now = time.time()
    expired = [k for k, exp in _hash_cache.items() if now >= exp]
    for k in expired:
        del _hash_cache[k]


def compute_changes_hash(changes: list) -> str:
    """
    对 changes 列表计算 MD5 hash。
    只取 new_path 和 diff 内容参与计算，忽略 additions/deletions 统计数字。
    """
    hasher = hashlib.md5()
    # 按文件路径排序，保证顺序无关
    for item in sorted(changes, key=lambda x: x.get("new_path", "")):
        path = item.get("new_path", "")
        diff = item.get("diff", "")
        hasher.update(f"{path}\n{diff}\n".encode("utf-8"))
    return hasher.hexdigest()


def is_duplicate(diff_hash: str) -> bool:
    """
    判断该 hash 是否在 TTL 内已经审查过。
    返回 True 表示重复，应跳过；False 表示首次或已过期，应继续审查。
    """
    if os.getenv("DIFF_DEDUP_ENABLED", "1") != "1":
        return False

    with _cache_lock:
        _cleanup_expired()
        now = time.time()
        expire = _hash_cache.get(diff_hash)
        if expire is not None and now < expire:
            return True
        return False


def mark_reviewed(diff_hash: str):
    """将该 hash 标记为已审查，写入缓存。"""
    if os.getenv("DIFF_DEDUP_ENABLED", "1") != "1":
        return

    with _cache_lock:
        _cleanup_expired()
        _hash_cache[diff_hash] = time.time() + _get_ttl()
        logger.debug(f"diff_cache: 记录 hash={diff_hash[:8]}，TTL={_get_ttl()}s，"
                     f"缓存条目数={len(_hash_cache)}")
