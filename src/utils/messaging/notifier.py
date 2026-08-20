from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.messaging.dingtalk import DingTalkNotifier
from src.utils.messaging.feishu import FeishuNotifier
from src.utils.messaging.webhook import ExtraWebhookNotifier
from src.utils.messaging.wecom import WeComNotifier


def send_notification(content, msg_type='text', title="通知", is_at_all=False, project_name=None, url_slug=None,
                      webhook_data: dict={}):
    """
    并发发送通知消息到配置的平台（钉钉、企业微信、飞书、自定义 Webhook）。
    各渠道的 HTTP 请求并发执行，总耗时取决于最慢的那个渠道，而非各渠道之和。

    :param content: 消息内容
    :param msg_type: 消息类型，支持 text 和 markdown
    :param title: 消息标题（markdown 类型时使用）
    :param is_at_all: 是否 @所有人
    :param project_name: 项目名称，用于按项目路由 Webhook
    :param url_slug: 由 Git 服务器 URL 转换成的 slug，用于按实例路由 Webhook
    :param webhook_data: push event、merge event 的原始数据
    """
    common_kwargs = dict(
        content=content, msg_type=msg_type, title=title,
        is_at_all=is_at_all, project_name=project_name, url_slug=url_slug,
    )
    system_data = {**common_kwargs, "webhook_data": webhook_data}

    def _send_dingtalk():
        DingTalkNotifier().send_message(**common_kwargs)

    def _send_wecom():
        WeComNotifier().send_message(**common_kwargs)

    def _send_feishu():
        FeishuNotifier().send_message(**common_kwargs)

    def _send_extra_webhook():
        ExtraWebhookNotifier().send_message(
            system_data={k: v for k, v in system_data.items() if k != "webhook_data"},
            webhook_data=webhook_data,
        )

    tasks = [_send_dingtalk, _send_wecom, _send_feishu, _send_extra_webhook]

    # 用线程池并发执行，各渠道互不阻塞
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {executor.submit(fn): fn.__name__ for fn in tasks}
        for future in as_completed(futures):
            name = futures[future]
            exc = future.exception()
            if exc:
                # 单个渠道失败不影响其他渠道
                from src.utils.log import logger
                logger.error(f"通知渠道 {name} 发送失败: {exc}")