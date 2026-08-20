import abc
import functools
import os
import re
import threading
from typing import Dict, Any, List

import yaml
from jinja2 import Template

from src.llm.factory import Factory
from src.utils.log import logger
from src.utils.token_util import count_tokens, truncate_text_by_tokens

# ---------------------------------------------------------------------------
# 进程级 LLM 客户端单例
# 同一个进程（RQ Worker 或子进程）在处理多个任务时复用同一个客户端实例，
# 避免重复初始化 HTTP Session / 连接池。
# 使用锁保证多线程环境下的安全初始化。
# ---------------------------------------------------------------------------
_llm_client_lock = threading.Lock()
_llm_client = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        with _llm_client_lock:
            if _llm_client is None:
                _llm_client = Factory.getClient()
    return _llm_client


# ---------------------------------------------------------------------------
# 提示词模板缓存
# 用文件的修改时间（mtime）作为缓存 key 的一部分。
# 当 prompt_templates.yml 被修改后，mtime 发生变化，lru_cache 会产生新的
# cache key，旧缓存自动失效，无需重启服务即可加载新提示词。
# ---------------------------------------------------------------------------
_PROMPT_FILE = "config/prompt_templates.yml"


def _get_prompt_file_mtime() -> float:
    """获取提示词文件的修改时间，文件不存在时返回 0。"""
    try:
        return os.path.getmtime(_PROMPT_FILE)
    except OSError:
        return 0.0


@functools.lru_cache(maxsize=128)
def _load_prompt_template_cached(prompt_key: str, style: str, _mtime: float) -> tuple:
    """
    从 config/prompt_templates.yml 加载并渲染提示词，结果按 (prompt_key, style, mtime) 缓存。
    mtime 变化时（文件被修改）自动使用新内容，无需重启。
    返回 (system_prompt_str, user_prompt_str) 元组。
    """
    with open(_PROMPT_FILE, "r", encoding="utf-8") as file:
        all_prompts = yaml.safe_load(file)

    prompts = all_prompts.get(prompt_key)
    if not prompts:
        raise KeyError(f"提示词配置中找不到 key: {prompt_key}")

    def render(template_str: str) -> str:
        return Template(template_str).render(style=style)

    return render(prompts["system_prompt"]), render(prompts["user_prompt"])


def load_prompt_template(prompt_key: str, style: str) -> tuple:
    """
    加载提示词模板的公开接口。
    自动传入当前文件 mtime，实现热更新感知缓存。
    """
    return _load_prompt_template_cached(prompt_key, style, _get_prompt_file_mtime())


class BaseReviewer(abc.ABC):
    """代码审查基类"""

    def __init__(self, prompt_key: str):
        # 使用进程级单例客户端，避免重复初始化
        self.client = _get_llm_client()
        self.prompts = self._load_prompts(prompt_key, os.getenv("REVIEW_STYLE", "professional"))

    def _load_prompts(self, prompt_key: str, style="professional") -> Dict[str, Any]:
        """加载提示词配置（支持热更新，文件修改后自动生效）"""
        try:
            system_prompt, user_prompt = load_prompt_template(prompt_key, style)
            return {
                "system_message": {"role": "system", "content": system_prompt},
                "user_message": {"role": "user", "content": user_prompt},
            }
        except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
            logger.error(f"加载提示词配置失败: {e}")
            raise Exception(f"提示词配置加载失败: {e}")

    def call_llm(self, messages: List[Dict[str, Any]]) -> str:
        """调用 LLM 进行代码审核"""
        logger.debug(f"向 AI 发送代码 Review 请求, messages: {messages}")
        logger.info("正在调用 LLM 进行代码审查...")
        review_result = self.client.completions(messages=messages)
        logger.debug(f"收到 AI 返回结果: {review_result}")
        logger.info("LLM 审查完成")
        return review_result

    @abc.abstractmethod
    def review_code(self, *args, **kwargs) -> str:
        """抽象方法，子类必须实现"""
        pass


class CodeReviewer(BaseReviewer):
    """代码 Diff 级别的审查"""

    def __init__(self):
        # 使用进程级单例客户端，避免重复初始化
        self.client = _get_llm_client()
        # 语言到提示词映射
        self.language_prompts = {
            'python': 'python_review_prompt',
            'javascript': 'javascript_review_prompt',
            'typescript': 'javascript_review_prompt',
            'java': 'java_review_prompt',
            'go': 'go_review_prompt',
            'php': 'php_review_prompt',
            'cpp': 'cpp_review_prompt',
            'c': 'cpp_review_prompt',
            'vue': 'vue3_review_prompt',
            'js': 'javascript_review_prompt',
            'ts': 'javascript_review_prompt',
            'py': 'python_review_prompt',
        }

    def _detect_language_from_diff(self, diffs_text: str) -> str:
        """从diff文本中检测主要编程语言"""
        # 文件扩展名到语言的映射
        file_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.vue': 'vue',
            '.java': 'java',
            '.go': 'go',
            '.php': 'php',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'cpp',
            '.hpp': 'cpp',
        }
        
        # 统计各种语言的文件数量
        language_counts = {}
        vue3_indicators = 0
        
        # 尝试多种文件路径模式
        file_patterns = [
            r'^\+\+\+ b/(.+)$',  # Git diff格式: +++ b/file.vue
            r'^\+\+\+ (.+)$',    # 简化格式: +++ file.vue
            r'^--- a/(.+)$',     # Git diff格式: --- a/file.vue
            r'^--- (.+)$',       # 简化格式: --- file.vue
            r'^diff --git a/(.+) b/(.+)$',  # Git diff格式: diff --git a/file.vue b/file.vue
        ]
        
        for line in diffs_text.split('\n'):
            # 尝试所有文件路径模式
            file_path = None
            for pattern in file_patterns:
                match = re.search(pattern, line)
                if match:
                    if pattern == r'^diff --git a/(.+) b/(.+)$':
                        # 对于diff --git格式，取第一个文件路径
                        file_path = match.group(1)
                    else:
                        file_path = match.group(1)
                    break
            
            if file_path:
                # 获取文件扩展名
                ext = os.path.splitext(file_path)[1].lower()
                if ext in file_extensions:
                    lang = file_extensions[ext]
                    language_counts[lang] = language_counts.get(lang, 0) + 1
                    logger.debug(f"检测到文件: {file_path}, 扩展名: {ext}, 语言: {lang}")
            
            # 检查代码内容中的Vue3特征
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in [
                'setup()', 'defineprops', 'defineemits', 'ref(', 'reactive(',
                'computed(', 'watch(', 'onmounted', 'onunmounted',
                'composition api', 'script setup', '<script setup'
            ]):
                vue3_indicators += 1
                logger.debug(f"检测到Vue3特征: {line.strip()}")
        
        # 如果没有通过文件路径检测到语言，尝试从内容中检测
        if not language_counts:
            # 检查是否包含Vue相关的内容
            if any(indicator in diffs_text.lower() for indicator in [
                '<template>', '<script>', '<style>', 'vue', '.vue'
            ]):
                language_counts['vue'] = 1
                logger.debug("通过内容检测到Vue文件")
            
            # 检查JavaScript特征
            elif any(indicator in diffs_text.lower() for indicator in [
                'function', 'var ', 'let ', 'const ', '=>', 'prompt(', 'alert(',
                'console.log', 'document.', 'window.', 'addEventListener'
            ]):
                language_counts['javascript'] = 1
                logger.debug("通过内容检测到JavaScript代码")
            
            # 检查Python特征
            elif any(indicator in diffs_text.lower() for indicator in [
                'def ', 'import ', 'from ', 'class ', 'if __name__', 'print(',
                'self.', 'return ', 'try:', 'except:', 'with open('
            ]):
                language_counts['python'] = 1
                logger.debug("通过内容检测到Python代码")
        
        # 返回出现最多的语言，如果没有检测到则返回默认
        if language_counts:
            primary_language = max(language_counts, key=language_counts.get)
            
            # 如果是Vue且有Vue3特征，记录日志
            if primary_language == 'vue' and vue3_indicators > 0:
                logger.debug(f"检测到Vue3代码，Vue3特征数量: {vue3_indicators}")
            elif primary_language == 'vue':
                logger.debug(f"检测到Vue文件，但未发现Vue3特征，Vue3特征数量: {vue3_indicators}")
            
            logger.info(f"检测到主要编程语言: {primary_language}")
            return primary_language
        
        logger.info("未检测到特定编程语言，使用通用审查提示词")
        return 'default'

    def _get_appropriate_prompt(self, diffs_text: str) -> str:
        """根据代码内容选择合适的提示词"""
        detected_lang = self._detect_language_from_diff(diffs_text)
        prompt_key = self.language_prompts.get(detected_lang, 'vue3_review_prompt')
        
        logger.debug(f"语言检测结果: {detected_lang} -> 提示词: {prompt_key}")
        
        # 临时修复：强制Vue文件使用Vue3提示词
        if detected_lang == 'vue':
            return 'vue3_review_prompt'
        
        return prompt_key

    def _load_language_specific_prompts(self, prompt_key: str, style="professional") -> Dict[str, Any]:
        """加载语言特定的提示词配置（支持热更新）"""
        try:
            system_prompt, user_prompt = load_prompt_template(prompt_key, style)
            return {
                "system_message": {"role": "system", "content": system_prompt},
                "user_message": {"role": "user", "content": user_prompt},
            }
        except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
            logger.error(f"加载语言特定提示词配置失败: {e}")
            return self._load_fallback_prompts(style)

    def _load_fallback_prompts(self, style="professional") -> Dict[str, Any]:
        """加载通用提示词作为回退（支持热更新）"""
        try:
            system_prompt, user_prompt = load_prompt_template("code_review_prompt", style)
            return {
                "system_message": {"role": "system", "content": system_prompt},
                "user_message": {"role": "user", "content": user_prompt},
            }
        except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
            logger.error(f"加载通用提示词配置失败: {e}")
            raise Exception(f"提示词配置加载失败: {e}")

    def _convert_changes_to_diff_format(self, changes: list) -> str:
        """将changes列表转换为标准的diff格式"""
        if not changes:
            return ""
        
        diff_content = []
        for change in changes:
            # 处理不同的change格式
            if isinstance(change, dict):
                # GitLab API返回的格式
                if 'diff' in change:
                    diff_text = change['diff']
                    # 如果diff不包含文件路径信息，但有new_path，则添加标准diff头
                    if not diff_text.startswith('diff --git') and 'new_path' in change:
                        diff_text = f"diff --git a/{change['new_path']} b/{change['new_path']}\nindex 0000000..0000000 100644\n--- a/{change['new_path']}\n+++ b/{change['new_path']}\n{diff_text}"
                    diff_content.append(diff_text)
                elif 'new_path' in change and 'old_path' in change:
                    # 构建简单的diff格式
                    diff_content.append(f"diff --git a/{change['old_path']} b/{change['new_path']}")
                    if 'new_file' in change and change['new_file']:
                        diff_content.append(f"new file mode 100644")
                    elif 'deleted_file' in change and change['deleted_file']:
                        diff_content.append(f"deleted file mode 100644")
                    diff_content.append(f"--- a/{change['old_path']}")
                    diff_content.append(f"+++ b/{change['new_path']}")
                    if 'diff' in change:
                        diff_content.append(change['diff'])
            elif isinstance(change, str):
                # 如果已经是字符串格式，直接添加
                diff_content.append(change)
        
        return "\n".join(diff_content)

    def review_and_strip_code(self, changes_text: str, commits_text: str = "", changes_data: list = None) -> str:
        """
        Review判断changes_text超出取前REVIEW_MAX_TOKENS个token，超出则截断changes_text，
        调用review_code方法，返回review_result，如果review_result是markdown格式，则去掉头尾的```
        :param changes_text: 可以是字符串或列表格式的changes
        :param commits_text:
        :param changes_data: 原始的changes数据，用于语言检测
        :return:
        """
        # 保存原始的changes数据用于语言检测
        original_changes_data = changes_data
        
        # 如果changes_text是列表格式，转换为diff格式
        if isinstance(changes_text, list):
            changes_text = self._convert_changes_to_diff_format(changes_text)
        elif hasattr(changes_text, '__iter__') and not isinstance(changes_text, str):
            # 处理其他可迭代对象
            changes_text = self._convert_changes_to_diff_format(list(changes_text))
        
        # 如果changes为空,打印日志
        if not changes_text:
            logger.info("代码为空, diffs_text = %", str(changes_text))
            return "代码为空"

        # 在截断之前先进行语言检测，确保能正确识别文件类型
        detected_language = self._detect_language_from_diff(changes_text)
        
        # 如果从diff中检测失败，尝试从changes数据中检测
        if detected_language == 'default' and original_changes_data:
            logger.debug("从diff中检测语言失败，尝试从changes数据中检测")
            detected_language = self._detect_language_from_changes(original_changes_data)
        
        # 如果超长，取前REVIEW_MAX_TOKENS个token
        # 优先使用环境变量中的显式配置；
        # 如果没有配置，则从当前 LLM 客户端查询模型上下文窗口，
        # 并预留 8000 tokens 给 system prompt、user prompt 模板和模型输出，
        # 剩余全部用于 diff 内容。
        env_max = os.getenv("REVIEW_MAX_TOKENS")
        if env_max:
            review_max_tokens = int(env_max)
        else:
            model_context = self.client.get_max_context_tokens()
            review_max_tokens = model_context - 8_000
            logger.info(f"REVIEW_MAX_TOKENS 未配置，自动使用模型上下文窗口: {model_context} - 8000 预留 = {review_max_tokens} tokens")
        
        # 计算tokens数量，如果超过REVIEW_MAX_TOKENS，截断changes_text
        tokens_count = count_tokens(changes_text)
        if tokens_count > review_max_tokens:
            logger.info(f"代码过长（{tokens_count} tokens），截断至 {review_max_tokens} tokens")
            changes_text = truncate_text_by_tokens(changes_text, review_max_tokens)
            # 截断后再次检测语言，以防截断破坏了文件路径信息
            truncated_language = self._detect_language_from_diff(changes_text)
            # 如果截断后检测不到语言，使用截断前的检测结果
            if truncated_language == 'default' and detected_language != 'default':
                logger.debug(f"截断后语言检测失败，沿用截断前结果: {detected_language}")
                final_language = detected_language
            else:
                final_language = truncated_language
        else:
            final_language = detected_language

        review_result = self.review_code(changes_text, commits_text, final_language, original_changes_data).strip()
        if review_result.startswith("```markdown") and review_result.endswith("```"):
            return review_result[11:-3].strip()
        return review_result

    def review_code(self, diffs_text: str, commits_text: str = "", pre_detected_language: str = None, changes_data: list = None) -> str:
        """Review 代码并返回结果"""
        # 智能选择提示词
        if pre_detected_language and pre_detected_language != 'default':
            detected_lang = pre_detected_language
        else:
            detected_lang = self._detect_language_from_diff(diffs_text)
            if detected_lang == 'default' and changes_data:
                detected_lang = self._detect_language_from_changes(changes_data)
        
        prompt_key = self.language_prompts.get(detected_lang, 'vue3_review_prompt')
        style = os.getenv("REVIEW_STYLE", "professional")
        
        logger.info(f"开始审查代码，语言: {detected_lang}，提示词: {prompt_key}，风格: {style}")
        
        # 加载对应的提示词
        if prompt_key != "code_review_prompt":
            try:
                prompts = self._load_language_specific_prompts(prompt_key, style)
            except Exception as e:
                logger.error(f"加载语言特定提示词失败: {e}, 回退到通用提示词")
                prompts = self._load_fallback_prompts(style)
        else:
            prompts = self._load_fallback_prompts(style)
        
        messages = [
            prompts["system_message"],
            {
                "role": "user",
                "content": prompts["user_message"]["content"].format(
                    diffs_text=diffs_text, commits_text=commits_text
                ),
            },
        ]
        return self.call_llm(messages)

    def _detect_language_from_changes(self, changes_data: list) -> str:
        """从changes数据中检测主要编程语言"""
        file_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.vue': 'vue',
            '.java': 'java',
            '.go': 'go',
            '.php': 'php',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'cpp',
            '.hpp': 'cpp',
        }
        
        language_counts = {}
        
        for change in changes_data:
            if isinstance(change, dict):
                # 尝试从new_path获取文件路径
                file_path = change.get('new_path') or change.get('old_path')
                if file_path:
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in file_extensions:
                        lang = file_extensions[ext]
                        language_counts[lang] = language_counts.get(lang, 0) + 1
                        logger.debug(f"从changes数据检测到文件: {file_path}, 语言: {lang}")
        
        if language_counts:
            primary_language = max(language_counts, key=language_counts.get)
            logger.debug(f"从changes数据检测到主要编程语言: {primary_language}")
            return primary_language
        
        logger.debug("从changes数据中未检测到特定编程语言")
        return 'default'

    @staticmethod
    def parse_review_score(review_text: str) -> int:
        """解析 AI 返回的 Review 结果，返回评分"""
        if not review_text:
            return 0
        match = re.search(r"总分[:：]\s*(\d+)分?", review_text)
        return int(match.group(1)) if match else 0

