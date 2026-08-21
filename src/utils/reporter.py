import os
import re
import yaml
from src.llm.factory import Factory
from src.utils.log import logger


def _load_user_map() -> dict:
    """
    从 config/user_map.yml 加载用户映射表。
    返回 {git用户名: 企微userid} 的字典，加载失败时返回空字典。
    """
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'user_map.yml')
    config_path = os.path.normpath(config_path)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data.get('wecom_user_map', {}) or {}
    except FileNotFoundError:
        logger.warning(f"未找到用户映射配置文件: {config_path}")
        return {}
    except Exception as e:
        logger.warning(f"加载用户映射配置失败: {e}")
        return {}


def _replace_authors_with_at_tags(report: str, authors: list) -> str:
    """
    将报告文本中出现的 author 名称替换为 <@企微userid> 格式。
    按名称长度降序替换，避免短名称误匹配长名称的子串。
    找不到映射时用原始名字兜底，即 <@原始名字>。
    """
    user_map = _load_user_map()
    # 按名称长度降序，避免 "lyh" 先于 "lyh_admin" 被替换
    sorted_authors = sorted(set(authors), key=len, reverse=True)
    for author in sorted_authors:
        if not author:
            continue
        wecom_id = user_map.get(author.strip(), author.strip())
        at_tag = f'<@{wecom_id}>'
        # 用词边界匹配，避免误替换包含该名字的其他词
        report = re.sub(r'(?<![<\w])' + re.escape(author) + r'(?![\w>])', at_tag, report)
    return report


class Reporter:
    def __init__(self):
        self.client = Factory().getClient()

    def generate_report(self, data: str, authors: list = None) -> str:
        """
        根据提交记录数据生成日报。
        :param data: JSON 格式的提交记录字符串
        :param authors: 提交人列表，用于在报告中将名称替换为 <@企微userid> 格式
        """
        report = self.client.completions(
            messages=[
                {"role": "user", "content": f"下面是以json格式记录员工代码提交信息。请总结这些信息，生成每个员工的工作日报摘要。员工姓名直接用json内容中的author属性值。特别要求:以Markdown格式返回。\n{data}"},
            ],
        )
        if authors:
            report = _replace_authors_with_at_tags(report, authors)
        return report
