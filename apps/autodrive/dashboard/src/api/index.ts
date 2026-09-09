/**
 * api/index.ts — API 服务层
 *
 * 封装所有 HTTP 请求和 WebSocket 连接。
 * 保留 Paddle 前端的调用形状，由后端兼容层读取 KernelLeaf V11 JSONL run。
 */

// ============================================================================
// HTTP 基础请求
// ============================================================================
async function request<T = any>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!response.ok) {
    const errBody = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(errBody.detail || `HTTP ${response.status}`)
  }
  return response.json()
}

// ============================================================================
// 训练相关 API
// ============================================================================
export const trainApi = {
  /** GET /api/train/status — 获取训练状态 */
  getStatus: () => request('/api/train/status'),
  /** GET /api/train/history — 获取训练历史 */
  getHistory: () => request('/api/train/history'),
  /** GET /api/train/config — 获取训练配置 */
  getConfig: () => request('/api/train/config'),
  start: (config: Record<string, any>) =>
    request('/api/train/start', { method: 'POST', body: JSON.stringify(config) }),
  stop: () => request('/api/train/stop', { method: 'POST' }),
  pause: () => request('/api/train/pause', { method: 'POST' }),
  resume: () => request('/api/train/resume', { method: 'POST' }),
}

// ============================================================================
// 系统资源 API
// ============================================================================
export const systemApi = {
  /** GET /api/system/current — 瞬时系统资源 */
  getCurrent: () => request('/api/system/current'),
  /** GET /api/system/history — 系统资源历史 */
  getHistory: (limit = 300) => request(`/api/system/history?limit=${limit}`),
}

// ============================================================================
// 模型相关 API
// ============================================================================
export const modelApi = {
  /** GET /api/model/info — 模型信息 */
  getInfo: () => request('/api/model/info'),
  /** GET /api/model/checkpoints — 检查点列表 */
  getCheckpoints: () => request('/api/model/checkpoints'),
}

// ============================================================================
// Grad-CAM 可视化 API
// ============================================================================
export const vizApi = {
  generateGradCam: (_config: Record<string, any>) =>
    Promise.reject(new Error('请使用命令行生成 Grad-CAM，再从历史记录加载')),
  /** GET /api/viz/list — 热力图列表 */
  getList: () => request('/api/viz/list'),
}

// ============================================================================
// 验证评估 API
// ============================================================================
export const evalApi = {
  /** GET /api/eval/validation — 读取训练时保存的最新验证结果 */
  validate: (_config: Record<string, any>) => request('/api/eval/validation'),
}

// ============================================================================
// 数据管理 API
// ============================================================================
export const dataApi = {
  /** GET /api/data/info — 数据集信息 */
  getInfo: () => request('/api/data/info'),
  regenerate: (_config: Record<string, any>) =>
    Promise.reject(new Error('请使用命令行重新生成 manifest')),
}

// ============================================================================
// 健康检查
// ============================================================================
export const healthApi = {
  /** GET /api/health — 健康检查 */
  check: () => request('/api/health'),
}

// ============================================================================
// WebSocket 连接
// ============================================================================
export type WsMessageHandler = (data: any) => void

export interface WsConnection {
  send: (data: string) => void
  close: () => void
}

/**
 * 建立 WebSocket 连接
 * @param onMessage - 消息处理回调
 * @param onOpen - 连接成功回调
 * @param onClose - 断连回调
 * @returns WsConnection 控制对象
 */
export function connectWebSocket(
  onMessage: WsMessageHandler,
  onOpen?: () => void,
  onClose?: () => void,
): WsConnection {
  // 开发模式直连后端 WebSocket（不走 Vite 代理，代理对 WS 升级不稳定）
  const wsUrl = import.meta.env.DEV
    ? 'ws://localhost:8000/ws/train'
    : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/train`

  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let reconnectDelay = 1000
  let closed = false

  function connect() {
    if (closed) return
    ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      reconnectDelay = 1000 // 重置退避
      onOpen?.()
      // 发送 ping 获取完整状态
      ws?.send('ping')
    }

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data)
        onMessage(data)
      } catch {
        // ignore parse errors
      }
    }

    ws.onclose = () => {
      if (closed) return
      onClose?.()
      // 指数退避重连（1s → 2s → 4s → 8s → max 30s）
      reconnectTimer = setTimeout(() => {
        reconnectDelay = Math.min(reconnectDelay * 2, 30000)
        connect()
      }, reconnectDelay)
    }

    ws.onerror = () => {
      ws?.close()
    }
  }

  connect()

  return {
    send: (data: string) => {
      if (ws?.readyState === WebSocket.OPEN) ws.send(data)
    },
    close: () => {
      closed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      ws?.close()
    },
  }
}
