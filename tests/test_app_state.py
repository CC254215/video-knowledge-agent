from app.models import VideoMetadata
from app.ui.state import AppState


def test_switching_video_clears_chat_history():
    state = AppState()
    state.chat_history.append({"role": "user", "content": "old"})
    old_conversation = state.current_conversation_id

    state.load_video("v1", metadata=VideoMetadata(video_id="v1", title="Video"))

    assert state.current_video_id == "v1"
    assert state.chat_history == []
    assert state.current_conversation_id != old_conversation


def test_clear_video_resets_current_video_id():
    state = AppState()
    state.load_video("v1")
    state.clear_video()

    assert state.current_video_id is None
    assert state.processing_status == "等待输入"
    assert state.chat_history == []
