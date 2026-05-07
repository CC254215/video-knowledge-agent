# Video Knowledge Agent MVP

Video Knowledge Agent 鏄竴涓湰鍦颁紭鍏堢殑瑙嗛鐭ヨ瘑 Agent銆傚畠涓嶆槸鏅€氳棰戞憳瑕佸伐鍏凤紝鑰屾槸鎶婅棰戝鐞嗘垚鍙拷婧€佸彲闂瓟銆佸彲瀵煎嚭鍒?Obsidian 鐨勭煡璇嗚瘉鎹€?
褰撳墠闃舵涓婚摼璺彧浣跨敤涓夌被璇佹嵁锛?
- `speech`锛氬瓧骞曟垨 ASR 鍙拌瘝鏂囨湰
- `frame`锛氬叧閿抚鍥剧墖
- `frame_caption`锛氬叧閿抚瑙嗚鎻忚堪

OCR 宸叉殏鏃剁鐢ㄣ€?
## 涓诲叆鍙?
鏈湴 Web UI锛?
```powershell
py -3.11 cli.py
```

鍙€夊弬鏁帮細

```powershell
py -3.11 cli.py --port 7861
py -3.11 cli.py --share
py -3.11 cli.py --debug
```

榛樿鍦板潃锛?
```text
http://127.0.0.1:7860
```

寮€鍙戣皟璇?CLI 浠嶇劧淇濈暀锛?
```powershell
py -3.11 -m src.cli process --url "<video_url>" --no-ocr --strict-llm
py -3.11 -m src.cli ask --video-id "<video_id>" --question "杩欎釜瑙嗛鏍稿績瑙傜偣鏄粈涔堬紵"
py -3.11 -m src.cli rebuild-index --video-id "<video_id>"
py -3.11 -m src.cli refine-frames --video-id "<video_id>" --start 600 --end 720
```

## 鐜瑕佹眰

- Python 3.11+
- ffmpeg 蹇呴』瀹夎骞跺姞鍏?PATH锛岀湡瀹?URL 澶勭悊渚濊禆瀹冧笅杞藉拰鍚堝苟瑙嗛/闊抽
- yt-dlp
- 鍙€夛細faster-whisper锛岀敤浜庢棤瀛楀箷瑙嗛 ASR

鏈」鐩篃浼氳嚜鍔ㄨ瘑鍒」鐩唴缃矾寰勶細

```text
D:\video_knowledge_agent\tools\ffmpeg\bin
```

瀹夎渚濊禆锛?
```powershell
py -3.11 -m pip install -e .
```

娉ㄦ剰锛氬鏋滅郴缁熼粯璁?`python` 鎸囧悜 Python 3.10锛岃浣跨敤 `py -3.11` 杩愯鏈」鐩€?
## .env 閰嶇疆

鎺ㄨ崘閰嶇疆锛?
```env
LLM_PROVIDER=zhipu
LLM_API_KEY=${ZHIPU_API_KEY}
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_SUMMARY_MODEL=glm-4.6

VLM_PROVIDER=zhipu
VLM_API_KEY=${ZHIPU_API_KEY}
VLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
VLM_CAPTION_MODEL=GLM-4.6V-FlashX

EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=Embedding-3

DATA_DIR=./data
OBSIDIAN_VAULT_PATH=
ASR_MODEL=small

ENABLE_OCR=false
EVIDENCE_TYPES=speech,frame,frame_caption
STRICT_LLM=false
ALLOW_CODING_ENDPOINT_FOR_RUNTIME=false
REQUEST_TIMEOUT_SECONDS=600
SUMMARY_MAX_SEGMENTS=24
SUMMARY_SEGMENT_CHARS=500
SUMMARY_CAPTION_CHARS=300
STORYLINE_TOP_K=16
```

## MemPalace MCP stdio 集成

当前 Agent 可以通过 MemPalace 原生 MCP server 与长期记忆系统通信。该集成默认关闭，不影响视频处理和问答主链路。
`.env` 示例：
```env
MEMPALACE_PROVIDER=mcp_stdio
MEMPALACE_COMMAND=python
MEMPALACE_ARGS=-m mempalace.mcp_server
MEMPALACE_PALACE_PATH=
MEMPALACE_TIMEOUT_SECONDS=30
MEMPALACE_AUTO_STATUS=true
```

如果需要指定 palace 路径：
```env
MEMPALACE_PALACE_PATH=D:\path\to\palace
```

调试命令：
```powershell
py -3.11 scripts\smoke_test_mempalace_mcp.py
py -3.11 -m app.cli mempalace-status
py -3.11 -m app.cli mempalace-search "agent memory"
```

使用边界：
- 当前视频事实仍必须由 `speech` / `frame` / `frame_caption` 证据支持。
- MemPalace 只作为长期上下文、用户偏好、历史洞察补充。
- MemPalace 检索结果会被结构化为 `long_term_memory`，不会直接拼接成视频证据。

鏅€氳棰戞憳瑕佸拰 Storyline 涓嶅簲浣跨敤锛?
```text
https://open.bigmodel.cn/api/coding/paas/v4
```

濡傛灉妫€娴嬪埌 runtime LLM 浣跨敤 coding endpoint锛岀郴缁熶細鎵撳嵃寮鸿鍛婏紱`STRICT_LLM=true` 鏃朵細鐩存帴閫€鍑恒€?
## 閰嶇疆妫€鏌ヤ笌 Smoke Test

妫€鏌ュ綋鍓嶅疄闄呰鍙栧埌鐨勯厤缃細

```powershell
py -3.11 scripts\check_runtime_config.py
```

娴嬭瘯鏂囨湰鎽樿妯″瀷锛?
```powershell
py -3.11 scripts\smoke_test_llm.py
```

娴嬭瘯瑙嗚 caption 妯″瀷锛?
```powershell
py -3.11 scripts\smoke_test_vlm.py
```

杩欎簺鑴氭湰浼氭墦鍗?provider銆乥ase URL銆乵odel銆丠TTP status 鍜屽搷搴旀鏂囷紝浣嗕笉浼氭墦鍗?API key銆?
## 澶勭悊鐪熷疄 URL

```powershell
py -3.11 -m src.cli process --url "<鍏紑瑙嗛 URL>" --no-ocr --strict-llm
```

鎴愬姛鏃惰緭鍑哄寘鍚細

- Pipeline Status
- LLM Summary: success
- Evidence Mode: speech + frame_caption, OCR disabled
- ChromaDB indexed: speech=N, frame_caption=M
- 30 绉掗€熻
- Structured Outline
- Query-Guided Storyline

杈撳嚭鏂囦欢淇濆瓨鍦細

```text
data/videos/{video_id}/
```

鏍稿績鏂囦欢锛?
- `metadata.json`
- `transcript.json`
- `multimodal_segments.json`
- `summary.json`
- `storyline.json`
- `llm_logs/`

## 鍗曡棰戦棶绛?
濡傛灉杩涚▼閲嶅惎锛屽厛閲嶅缓鍐呭瓨鍚戦噺绱㈠紩锛?
```powershell
py -3.11 -m src.cli rebuild-index --video-id "<video_id>"
```

鐒跺悗鎻愰棶锛?
```powershell
py -3.11 -m src.cli ask --video-id "<video_id>" --question "杩欎釜瑙嗛鏍稿績瑙傜偣鏄粈涔堬紵"
```

鍥炵瓟蹇呴』鍖呭惈锛?
- answer
- evidence
- timestamps
- evidence_types
- confidence
- agreement_score
- needs_visual_check
- reason

璇佹嵁涓嶈冻鏃讹紝Agent 蹇呴』鏄庣‘璇存槑褰撳墠瑙嗛璇佹嵁涓嶈冻锛屼笉鑳界紪閫犮€?
## Obsidian 瀵煎嚭

閰嶇疆 `OBSIDIAN_VAULT_PATH` 鍚庯紝澶勭悊瀹屾垚浼氬皾璇曞鍑猴細

- `10_Sources/Videos/{date} - {safe_title}.md`
- `30_Storylines/{date} - {safe_title} - storyline.md`

濡傛灉鏈厤缃?vault锛岀郴缁熶細璺宠繃瀵煎嚭锛屼笉闃诲涓绘祦绋嬨€?
## 褰撳墠闄愬埗

- 骞冲彴鏀寔渚濊禆 yt-dlp锛岀綉绔欑瓥鐣ュ彉鍖栧彲鑳藉鑷村け鏁堛€?- 鐪熷疄 URL 澶勭悊闇€瑕?ffmpeg 鍦?PATH 涓彲鐢ㄣ€?- 绗竴鐗堣瑙夌悊瑙ｅ彧浣跨敤鍏抽敭甯?caption锛屼笉鍋氬畬鏁村妯℃€佽瑙夋帹鐞嗐€?- OCR 褰撳墠绂佺敤銆?- ASR 璐ㄩ噺渚濊禆鏈湴妯″瀷鍜岄煶棰戣川閲忋€?- ChromaDB 褰撳墠浣跨敤 persistent collection锛岄粯璁よ矾寰?`data/chroma/`锛沗rebuild-index` 浠嶄繚鐣欑敤浜庝粠 JSON 閲嶅缓绱㈠紩銆?- DPP 鏄瘉鎹€夋嫨鍜屼竴鑷存€т俊鍙凤紝涓嶆槸鐪熶吉璇佹槑銆?
## Rate Limit 涓庢柇鐐规仮澶?
褰撳墠 pipeline 淇濈暀 summary銆乻toryline銆乂LM caption 绛夊唴閮ㄥ苟琛岋紝浣嗘墍鏈夊閮ㄤ緷璧栬姹傞兘浼氱粡杩囧垎璧勬簮璋冨害鍣細

- `llm`锛氭枃鏈憳瑕併€丼toryline銆侀棶绛旂敓鎴?- `vlm`锛氬叧閿抚 caption
- `embedding`锛氬悜閲忓寲
- `ytdlp`锛氳棰戠綉绔?metadata銆佸瓧骞曘€侀煶瑙嗛涓嬭浇
- `asr`锛氫簯绔?ASR 鎺ュ彛锛屾湰鍦?faster-whisper 涓嶈蛋 API 闄愭祦

鍙湪 `.env` 涓皟鑺傦細

```env
LLM_MAX_CONCURRENCY=2
LLM_MIN_INTERVAL_SECONDS=1.5
VLM_CAPTION_CONCURRENCY=3
VLM_MIN_INTERVAL_SECONDS=2.0
EMBEDDING_MAX_CONCURRENCY=1
EMBEDDING_MIN_INTERVAL_SECONDS=1.0
YTDLP_MAX_CONCURRENCY=1
YTDLP_MIN_INTERVAL_SECONDS=6.0
ASR_MAX_CONCURRENCY=1
ASR_MIN_INTERVAL_SECONDS=2.0
API_MAX_RETRIES=3
API_RETRY_BASE_DELAY_SECONDS=2.0
PIPELINE_RESUME=true
PIPELINE_FORCE_STAGE=
```

姣忎釜瑙嗛鐩綍浼氬啓鍏?`processing_state.json`銆傞噸鏂拌繍琛?`process` 鏃堕粯璁よ烦杩囧凡鎴愬姛涓斾骇鐗╁瓨鍦ㄧ殑闃舵锛?
```powershell
py -3.11 -m app.cli process --url "<video_url>"
```

寮哄埗鍏ㄩ儴閲嶈窇锛?
```powershell
py -3.11 -m app.cli process --url "<video_url>" --force-refresh
```

鍙噸璺戞煇涓€闃舵锛?
```powershell
py -3.11 -m app.cli process --url "<video_url>" --force-stage summary
```

绂佺敤鏂偣鎭㈠锛?
```powershell
py -3.11 -m app.cli process --url "<video_url>" --no-resume
```
