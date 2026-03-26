#!/usr/bin/env bash
#
# smoke-test.sh — 一键启动 Pasloe + Trenni，提交测试任务，观察结果
#
# 用法:
#   ./scripts/smoke-test.sh                          # 使用默认 Anthropic API
#   ./scripts/smoke-test.sh --api-base URL --api-key KEY --api-key-env ENV_NAME
#
# 前置条件:
#   - pip install -e ../pasloe .       # pasloe 和 trenni 已安装
#   - 设置 LLM API key 环境变量（默认 ANTHROPIC_API_KEY）
#   - 本地已构建 localhost/yoitsu-palimpsest-job:dev
#
set -euo pipefail

# ── 颜色 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── 默认值 ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PASLOE_PORT=18900
PASLOE_HOST=127.0.0.1
WORK_DIR="${PROJECT_DIR}/smoke-work"
TASK="List the files in the evo repository and describe what each role does. Then stop calling tools and summarize the result."
JOB_IMAGE="localhost/yoitsu-palimpsest-job:dev"
PODMAN_SOCKET_URI="${PODMAN_HOST:-unix://${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/podman/podman.sock}"

# LLM 配置（可通过参数覆盖）
API_BASE=""
API_KEY=""
API_KEY_ENV="ANTHROPIC_API_KEY"
MODEL="claude-sonnet-4-6"

# 工作区配置
WORK_REPO=""
WORK_BRANCH="main"

# 超时
JOB_TIMEOUT=120   # 等待 job 完成的最大秒数

# ── 参数解析 ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-base)    API_BASE="$2";    shift 2;;
        --api-key)     API_KEY="$2";     shift 2;;
        --api-key-env) API_KEY_ENV="$2"; shift 2;;
        --model)       MODEL="$2";       shift 2;;
        --repo)        WORK_REPO="$2";   shift 2;;
        --branch)      WORK_BRANCH="$2"; shift 2;;
        --task)        TASK="$2";        shift 2;;
        --timeout)     JOB_TIMEOUT="$2"; shift 2;;
        --port)        PASLOE_PORT="$2"; shift 2;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --api-base URL       LLM API base URL (e.g. https://api.anthropic.com)"
            echo "  --api-key KEY        LLM API key value"
            echo "  --api-key-env NAME   Env var name for API key (default: ANTHROPIC_API_KEY)"
            echo "  --model MODEL        LLM model name (default: claude-sonnet-4-6)"
            echo "  --repo URL           Git repo URL to work on (optional)"
            echo "  --branch BRANCH      Git branch (default: main)"
            echo "  --task TEXT          Task to submit"
            echo "  --timeout SECS       Max seconds to wait for job (default: 120)"
            echo "  --port PORT          Pasloe port (default: 18900)"
            exit 0;;
        *) fail "Unknown option: $1"; exit 1;;
    esac
done

# 如果传了 --api-key，注入到对应环境变量
if [[ -n "$API_KEY" ]]; then
    export "$API_KEY_ENV"="$API_KEY"
fi

# ── 前置检查 ───────────────────────────────────────────────────────────
info "前置检查..."

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        fail "找不到命令: $1 (请先 pip install)"
        exit 1
    fi
}
check_cmd trenni
check_cmd uvicorn
check_cmd curl
check_cmd podman

if ! podman image exists "$JOB_IMAGE"; then
    fail "缺少 job image: $JOB_IMAGE"
    echo "请先执行: podman build -t $JOB_IMAGE -f deploy/podman/palimpsest-job.Containerfile ."
    exit 1
fi

if [[ -z "${!API_KEY_ENV:-}" ]]; then
    warn "环境变量 $API_KEY_ENV 未设置（LLM 调用会失败）"
    warn "用 --api-key 或 export $API_KEY_ENV=... 设置后重试"
fi

ok "前置检查通过"

# ── 清理函数 ───────────────────────────────────────────────────────────
PASLOE_PID=""
TRENNI_PID=""

cleanup() {
    info "清理进程..."
    [[ -n "$TRENNI_PID" ]] && kill "$TRENNI_PID" 2>/dev/null && wait "$TRENNI_PID" 2>/dev/null || true
    [[ -n "$PASLOE_PID" ]] && kill "$PASLOE_PID" 2>/dev/null && wait "$PASLOE_PID" 2>/dev/null || true
    info "清理完成"
}
trap cleanup EXIT

# ── 准备工作目录 ──────────────────────────────────────────────────────
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
PASLOE_DB="$WORK_DIR/events.db"

# ── 启动 Pasloe ──────────────────────────────────────────────────────
info "启动 Pasloe (port=$PASLOE_PORT)..."
SQLITE_PATH="$PASLOE_DB" ALLOW_INSECURE_HTTP=True \
    uvicorn pasloe.app:app --host "$PASLOE_HOST" --port "$PASLOE_PORT" \
    > "$WORK_DIR/pasloe.log" 2>&1 &
PASLOE_PID=$!

PASLOE_URL="http://${PASLOE_HOST}:${PASLOE_PORT}"
for i in $(seq 1 10); do
    if curl -sf "${PASLOE_URL}/health" >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

if ! curl -sf "${PASLOE_URL}/health" >/dev/null 2>&1; then
    fail "Pasloe 启动失败，查看日志: $WORK_DIR/pasloe.log"
    cat "$WORK_DIR/pasloe.log"
    exit 1
fi
ok "Pasloe 已启动 (PID=$PASLOE_PID)"

# ── 生成 Trenni 配置 ─────────────────────────────────────────────────
CONFIG_FILE="$WORK_DIR/trenni-config.yaml"

LLM_BLOCK=""
if [[ -n "$API_BASE" || "$API_KEY_ENV" != "ANTHROPIC_API_KEY" || "$MODEL" != "claude-sonnet-4-6" ]]; then
    LLM_BLOCK="default_llm:"
    [[ -n "$API_BASE" ]] && LLM_BLOCK="${LLM_BLOCK}
  api_base: \"${API_BASE}\""
    LLM_BLOCK="${LLM_BLOCK}
  api_key_env: \"${API_KEY_ENV}\"
  model: \"${MODEL}\"
  max_iterations: 20
  max_tokens: 4096"
fi

cat > "$CONFIG_FILE" <<YAML
pasloe_url: "${PASLOE_URL}"
runtime:
  kind: "podman"
  podman:
    socket_uri: "${PODMAN_SOCKET_URI}"
    pod_name: "yoitsu-dev"
    image: "${JOB_IMAGE}"
    pull_policy: "never"
    git_token_env: "GIT_TOKEN"
    env_allowlist:
      - "${API_KEY_ENV}"
max_workers: 2
poll_interval: 2.0
default_workspace:
  depth: 1
${LLM_BLOCK}
YAML

info "Trenni 配置:"
cat "$CONFIG_FILE" | sed 's/^/  /'

# ── 启动 Trenni ──────────────────────────────────────────────────────
info "启动 Trenni supervisor..."
trenni start --config "$CONFIG_FILE" > "$WORK_DIR/trenni.log" 2>&1 &
TRENNI_PID=$!
sleep 2

if ! kill -0 "$TRENNI_PID" 2>/dev/null; then
    fail "Trenni 启动失败，查看日志: $WORK_DIR/trenni.log"
    cat "$WORK_DIR/trenni.log"
    exit 1
fi
ok "Trenni 已启动 (PID=$TRENNI_PID)"

# ── 提交测试任务 ──────────────────────────────────────────────────────
info "提交测试任务..."
TASK_DATA=$(python3 -c "import json; print(json.dumps({
    'source_id': 'smoke-test',
    'type': 'task.submit',
    'data': {
        'task': '''$TASK''',
        'role': 'default',
        'repo': '''$WORK_REPO''',
        'branch': '''$WORK_BRANCH'''
    }
}))")
SUBMIT_RESP=$(curl -sf -X POST "${PASLOE_URL}/events" \
    -H "Content-Type: application/json" \
    -d "$TASK_DATA")

EVENT_ID=$(echo "$SUBMIT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
ok "任务已提交 (event_id=$EVENT_ID)"

# ── 等待 job 被 launch ───────────────────────────────────────────────
info "等待 supervisor 拾取任务..."
JOB_ID=""
for i in $(seq 1 20); do
    EVENTS=$(curl -sf "${PASLOE_URL}/events?order=asc" 2>/dev/null || echo "[]")
    JOB_ID=$(echo "$EVENTS" | python3 -c "
import sys, json
events = json.load(sys.stdin)
for e in events:
    if e['type'] == 'supervisor.job.launched':
        print(e['data']['job_id'])
        break
" 2>/dev/null || true)
    [[ -n "$JOB_ID" ]] && break
    sleep 1
done

if [[ -z "$JOB_ID" ]]; then
    fail "Supervisor 未启动 job，查看日志: $WORK_DIR/trenni.log"
    tail -20 "$WORK_DIR/trenni.log"
    exit 1
fi
ok "Job 已启动: $JOB_ID"

# ── 等待 job 完成 ────────────────────────────────────────────────────
info "等待 job 完成 (超时: ${JOB_TIMEOUT}s)..."
ELAPSED=0
FINAL_STATUS=""

while [[ $ELAPSED -lt $JOB_TIMEOUT ]]; do
    # 检查 trenni 是否还活着
    if ! kill -0 "$TRENNI_PID" 2>/dev/null; then
        warn "Trenni 进程已退出"
        break
    fi

# 查询 eventstore 看是否有终态事件
    EVENTS=$(curl -sf "${PASLOE_URL}/events?order=asc" 2>/dev/null || echo "[]")
    FINAL_STATUS=$(echo "$EVENTS" | python3 -c "
import sys, json
events = json.load(sys.stdin)
for e in events:
    if e['type'] in ('job.completed', 'job.failed') and e['data'].get('job_id') == '$JOB_ID':
        print(e['type'])
        break
" 2>/dev/null || true)
    [[ -n "$FINAL_STATUS" ]] && break

    sleep 3
    ELAPSED=$((ELAPSED + 3))
    # 每 15 秒输出一次进度
    if (( ELAPSED % 15 == 0 )); then
        info "已等待 ${ELAPSED}s..."
    fi
done

# ── 结果汇报 ──────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
info "测试结果汇报"
echo "════════════════════════════════════════════════════════════"

# 显示所有事件
info "Pasloe 事件流:"
curl -sf "${PASLOE_URL}/events?order=asc" | python3 -c "
import sys, json
events = json.load(sys.stdin)
for e in events:
    src = e.get('source_id', '?')
    typ = e['type']
    job = e.get('data', {}).get('job_id', '')
    extra = ''
    if typ == 'supervisor.job.launched':
        extra = f' container_id={e[\"data\"].get(\"container_id\",\"?\")} role={e[\"data\"].get(\"role\",\"?\")}'
    elif typ in ('job.completed', 'job.failed'):
        extra = f' summary={e[\"data\"].get(\"summary\",\"(none)\")[:80]}'
    print(f'  [{src}] {typ} {job}{extra}')
" 2>/dev/null || warn "无法获取事件"

echo ""

# 最终判定
if [[ "$FINAL_STATUS" == "job.completed" ]]; then
    ok "Job 完成！端到端测试通过"
    exit 0
elif [[ "$FINAL_STATUS" == "job.failed" ]]; then
    warn "Job 失败（但流程跑通了）"
    info "查看 trenni 日志: $WORK_DIR/trenni.log"
    exit 1
elif [[ $ELAPSED -ge $JOB_TIMEOUT ]]; then
    warn "等待超时 (${JOB_TIMEOUT}s)，job 可能仍在运行"
    info "查看 trenni 日志: $WORK_DIR/trenni.log"
    info "查看 pasloe 日志: $WORK_DIR/pasloe.log"
    exit 2
else
    warn "Job launch 成功但未产生终态事件"
    info "查看 trenni 日志最后 20 行:"
    tail -20 "$WORK_DIR/trenni.log" | sed 's/^/  /'
    exit 1
fi
