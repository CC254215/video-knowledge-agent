from app.config import Settings
from app.services.chat_service import ChatService


def test_chat_requires_video():
    result = ChatService(Settings(_env_file=None)).ask(None, "这个视频讲了什么？")

    assert "请先输入 URL 或上传视频并完成处理" in result.answer
    assert result.error == "missing_video"
    assert result.confidence == "low"
