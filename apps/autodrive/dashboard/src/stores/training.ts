/**
 * stores/training.ts — 训练状态 Pinia Store
 *
 * 管理所有训练相关状态，处理 WebSocket 实时消息。
 *
 * WebSocket 消息类型（由 KernelLeaf V11 兼容接口提供）：
 * 1. 训练状态快照 — 初始化 / ping 响应 / 每个 epoch 完成时推送
 *    { running, paused, epoch, total_epochs, steer_loss, throttle_loss,
 *      best_steer_loss, best_throttle_loss, lr, message, system, system_recent? }
 * 2. 系统资源独立推送 — 每 2 秒
 *    { type: "system", data: { cpu_percent, memory_percent, gpus, ... } }
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { trainApi, systemApi, modelApi, connectWebSocket, type WsConnection } from '@/api'

// ============================================================================
// 类型定义
// ============================================================================
export interface TrainHistoryEntry {
  epoch: number
  lr: number
  train_loss?: number | null
  val_loss?: number | null
  accuracy?: number | null
  steer_loss: number
  throttle_loss: number
  best_steer_loss: number
  best_throttle_loss: number
  epoch_time: number
  n_batches: number
  is_best_steer: boolean
  is_best_throttle: boolean
}

export interface GpuSnapshot {
  index: number
  name: string
  util_percent: number
  mem_used_mb: number
  mem_total_mb: number
  mem_percent: number
  temp_c: number
}

export interface SystemSnapshot {
  cpu_percent: number
  memory_percent: number
  memory_used_gb: number
  memory_total_gb: number
  gpus: GpuSnapshot[]
  disk_percent: number
  disk_used_gb: number
  disk_total_gb: number
}

export interface TrainConfig {
  manifest: string
  map: string
  batch_size: number
  epochs: number
  lr: number
  weight_decay: number
  grad_clip_norm: number
  device: string
}

export type TrainStatus = 'idle' | 'running' | 'paused'

// ============================================================================
// Store
// ============================================================================
export const useTrainingStore = defineStore('training', () => {
  // ----- 训练状态 -----
  const running = ref(false)
  const paused = ref(false)
  const epoch = ref(0)
  const totalEpochs = ref(100)
  const steerLoss = ref<number | null>(null)
  const throttleLoss = ref<number | null>(null)
  const bestSteerLoss = ref(Infinity)
  const bestThrottleLoss = ref(Infinity)
  const lr = ref(1e-4)
  const message = ref('idle')
  const startedAt = ref<number | null>(null)

  // ----- 训练历史（每个 epoch 一个条目） -----
  const history = ref<TrainHistoryEntry[]>([])

  // ----- 训练配置 -----
  const config = ref<TrainConfig>({
    manifest: 'data/DonkeyCar/manifest.jsonl',
    map: '',
    batch_size: 32,
    epochs: 10,
    lr: 1e-3,
    weight_decay: 1e-4,
    grad_clip_norm: 5.0,
    device: 'cpu',
  })

  // ----- 系统资源瞬时值 -----
  const system = ref<SystemSnapshot>({
    cpu_percent: 0,
    memory_percent: 0,
    memory_used_gb: 0,
    memory_total_gb: 15.95,
    gpus: [],
    disk_percent: 0,
    disk_used_gb: 0,
    disk_total_gb: 475.6,
  })

  // ----- 系统资源历史（用于 SystemChart 绘制曲线） -----
  // 每个元素包含时间戳，保留最近 10 分钟 = 300 条
  const systemHistory = ref<Array<SystemSnapshot & { timestamp: number }>>([])

  // ----- 模型信息 -----
  const modelInfo = ref<any>(null)
  const checkpoints = ref<any[]>([])

  // ----- WebSocket 连接 -----
  let wsConnection: WsConnection | null = null

  // ==========================================================================
  // 计算属性
  // ==========================================================================
  const status = computed<TrainStatus>(() => {
    if (!running.value) return 'idle'
    if (paused.value) return 'paused'
    return 'running'
  })

  const progressPercent = computed(() => {
    if (totalEpochs.value === 0) return 0
    return Math.min(100, Math.round((epoch.value / totalEpochs.value) * 100))
  })

  const currentGpu = computed<GpuSnapshot | null>(() =>
    system.value.gpus?.[0] || null,
  )

  // ==========================================================================
  // 初始化加载
  // ==========================================================================
  async function fetchInitialData() {
    console.log('[Store] 📡 开始加载初始数据...')
    try {
      const [statusData, historyData, sysHistoryData, modelData, checkpointData, configData] =
        await Promise.all([
          trainApi.getStatus(),
          trainApi.getHistory(),
          systemApi.getHistory(300),
          modelApi.getInfo(),
          modelApi.getCheckpoints(),
          trainApi.getConfig(),
        ])

      // 恢复训练状态
      console.log('[Store] 训练状态:', statusData)
      running.value = statusData.running ?? false
      paused.value = statusData.paused ?? false
      epoch.value = statusData.epoch ?? 0
      totalEpochs.value = statusData.total_epochs ?? 100
      steerLoss.value = statusData.steer_loss ?? null
      throttleLoss.value = statusData.throttle_loss ?? null
      bestSteerLoss.value = statusData.best_steer_loss ?? Infinity
      bestThrottleLoss.value = statusData.best_throttle_loss ?? Infinity
      lr.value = statusData.lr ?? 1e-4
      message.value = statusData.message ?? 'idle'
      startedAt.value = statusData.started_at ?? null

      // 恢复训练历史
      if (historyData.epochs) {
        history.value = historyData.epochs
        console.log(`[Store] 加载了 ${history.value.length} 条训练历史`)
      }

      // 恢复系统历史
      if (sysHistoryData.samples) {
        systemHistory.value = sysHistoryData.samples.map((s: any) => ({
          ...s,
          timestamp: s.timestamp || (Date.now() / 1000),
        }))
        console.log(`[Store] 加载了 ${systemHistory.value.length} 条系统历史`)
      }

      // 恢复系统瞬时值
      if (statusData.system) {
        system.value = { ...statusData.system }
      }

      // 恢复配置
      if (configData) {
        config.value = {
          manifest: configData.manifest ?? 'data/DonkeyCar/manifest.jsonl',
          map: configData.map ?? '',
          batch_size: configData.batch_size ?? 96,
          epochs: configData.epochs ?? 100,
          lr: configData.lr ?? 1e-4,
          weight_decay: configData.weight_decay ?? 1e-4,
          grad_clip_norm: configData.grad_clip_norm ?? 5.0,
          device: configData.device ?? 'auto',
        }
      }

      modelInfo.value = modelData
      checkpoints.value = checkpointData.checkpoints || []
    } catch (err: any) {
      console.error('[Store] 初始化数据加载失败:', err.message)
    }
  }

  // ==========================================================================
  // 训练控制
  // ==========================================================================
  async function startTraining(trainConfig?: Partial<TrainConfig>) {
    const cfg = { ...config.value, ...trainConfig }
    console.log('[Store] 🚀 发送开始训练请求:', cfg)
    const result = await trainApi.start(cfg)
    console.log('[Store] 训练已启动:', result)
    config.value = cfg
    running.value = true
    paused.value = false
    epoch.value = 0
    totalEpochs.value = cfg.epochs
    message.value = result.message || '正在启动'
    // 训练启动后，立即清空本地历史（等待 WS 推送新 epoch 数据）
    history.value = []
  }

  async function stopTraining() {
    console.log('[Store] ⏹ 发送停止请求')
    const result = await trainApi.stop()
    message.value = result.message || '正在停止'
  }

  async function pauseTraining() {
    console.log('[Store] ⏸ 发送暂停请求')
    const result = await trainApi.pause()
    paused.value = true
    message.value = result.message || '训练已暂停'
  }

  async function resumeTraining() {
    console.log('[Store] ▶ 发送恢复请求')
    const result = await trainApi.resume()
    paused.value = false
    message.value = result.message || '训练已恢复'
  }

  // ==========================================================================
  // 刷新训练历史（从 API 获取完整历史）
  // ==========================================================================
  async function fetchHistory() {
    try {
      const data = await trainApi.getHistory()
      if (data.epochs && data.epochs.length !== history.value.length) {
        history.value = data.epochs
        console.log(`[Store] 📋 训练历史已更新: ${history.value.length} 条`)
      }
    } catch { /* ignore */ }
  }

  // ==========================================================================
  // WebSocket 连接
  // ==========================================================================
  function connectWs() {
    // 已有连接则跳过，避免 close → reconnect → close 的死循环
    if (wsConnection) {
      console.log('[Store] 🔌 WebSocket 已连接，跳过重复连接')
      return
    }

    console.log('[Store] 🔌 正在建立 WebSocket 连接...')

    wsConnection = connectWebSocket(
      // ---- onMessage ----
      (data: any) => {
        // ---------- 消息类型 1：系统资源独立推送 ----------
        if (data.type === 'system') {
          const d = data.data
          console.log('[WS] system msg received:', d.cpu_percent + '% CPU,', d.gpus?.[0]?.util_percent + '% GPU')
          // 更新瞬时系统资源
          const gpus: GpuSnapshot[] = (d.gpus || []).map((g: any, i: number) => ({
            index: g.index ?? i,
            name: g.name || system.value.gpus[i]?.name || 'Unknown GPU',
            util_percent: g.util_percent ?? 0,
            mem_used_mb: g.mem_used_mb ?? 0,
            mem_total_mb: g.mem_total_mb ?? 8188,
            mem_percent: g.mem_percent ?? 0,
            temp_c: g.temp_c ?? 0,
          }))
          // 直接替换整个对象，触发 computed 重新计算
          system.value = {
            cpu_percent: d.cpu_percent ?? 0,
            memory_percent: d.memory_percent ?? 0,
            memory_used_gb: d.memory_used_gb ?? 0,
            memory_total_gb: d.memory_total_gb ?? 0,
            gpus,
            disk_percent: d.disk_percent ?? 0,
            disk_used_gb: d.disk_used_gb ?? 0,
            disk_total_gb: d.disk_total_gb ?? 0,
          }
          // 用替换代替 push，强制 Vue 检测数组变化
          const newEntry = {
            cpu_percent: system.value.cpu_percent,
            memory_percent: system.value.memory_percent,
            memory_used_gb: system.value.memory_used_gb,
            memory_total_gb: system.value.memory_total_gb,
            gpus: [...system.value.gpus],
            disk_percent: system.value.disk_percent,
            disk_used_gb: system.value.disk_used_gb,
            disk_total_gb: system.value.disk_total_gb,
            timestamp: d.timestamp || (Date.now() / 1000),
          }
          systemHistory.value = [...systemHistory.value, newEntry].slice(-300)
          return
        }

        // ---------- 消息类型 2：训练状态完整快照 ----------
        if (data.running !== undefined) {
          const wasRunning = running.value
          const wasPaused = paused.value

          running.value = data.running
          paused.value = data.paused ?? false
          epoch.value = data.epoch ?? 0
          totalEpochs.value = data.total_epochs ?? 100
          steerLoss.value = data.steer_loss ?? null
          throttleLoss.value = data.throttle_loss ?? null
          bestSteerLoss.value = data.best_steer_loss ?? Infinity
          bestThrottleLoss.value = data.best_throttle_loss ?? Infinity
          lr.value = data.lr ?? 1e-4
          message.value = data.message ?? ''

          // 更新系统资源（如果状态快照中包含）
          if (data.system) {
            system.value = {
              cpu_percent: data.system.cpu_percent ?? system.value.cpu_percent,
              memory_percent: data.system.memory_percent ?? system.value.memory_percent,
              memory_used_gb: data.system.memory_used_gb ?? system.value.memory_used_gb,
              memory_total_gb: data.system.memory_total_gb ?? system.value.memory_total_gb,
              gpus: (data.system.gpus || []).map((g: any, i: number) => ({
                index: g.index ?? i,
                name: g.name || system.value.gpus[i]?.name || 'Unknown GPU',
                util_percent: g.util_percent ?? 0,
                mem_used_mb: g.mem_used_mb ?? 0,
                mem_total_mb: g.mem_total_mb ?? 8188,
                mem_percent: g.mem_percent ?? 0,
                temp_c: g.temp_c ?? 0,
              })),
              disk_percent: data.system.disk_percent ?? system.value.disk_percent,
              disk_used_gb: data.system.disk_used_gb ?? system.value.disk_used_gb,
              disk_total_gb: data.system.disk_total_gb ?? system.value.disk_total_gb,
            }
          }

          // 恢复系统历史（初始连接时）
          if (data.system_recent && Array.isArray(data.system_recent)) {
            for (const s of data.system_recent) {
              systemHistory.value.push({
                ...s,
                timestamp: s.timestamp || (Date.now() / 1000),
              })
            }
            if (systemHistory.value.length > 300) {
              systemHistory.value = systemHistory.value.slice(-300)
            }
          }
          if (data.system_history && Array.isArray(data.system_history)) {
            for (const s of data.system_history) {
              systemHistory.value.push({
                ...s,
                timestamp: s.timestamp || (Date.now() / 1000),
              })
            }
            if (systemHistory.value.length > 300) {
              systemHistory.value = systemHistory.value.slice(-300)
            }
          }

          // 当训练状态变化时，立即刷新训练历史（获取最新的 epoch 数据）
          // 训练运行中：每个 WS 消息都可能代表新 epoch 完成
          if (data.running && data.epoch > 0) {
            fetchHistory()
          }

          // 日志
          if (!wasRunning && data.running) {
            console.log('[Store] 🟢 训练已启动')
          } else if (wasRunning && !data.running) {
            console.log('[Store] ⚪ 训练已结束')
          } else if (!wasPaused && data.paused) {
            console.log('[Store] 🟡 训练已暂停')
          } else if (wasPaused && !data.paused) {
            console.log('[Store] 🟢 训练已恢复')
          }
        }
      },
      // ---- onOpen ----
      () => {
        console.log('[Store] ✅ WebSocket 已连接')
      },
      // ---- onClose ----
      () => {
        console.log('[Store] ❌ WebSocket 已断开，将自动重连')
      },
    )
  }

  /** 清理 WebSocket 连接 */
  function dispose() {
    if (wsConnection) {
      wsConnection.close()
      wsConnection = null
    }
  }

  return {
    // state
    running, paused, epoch, totalEpochs, steerLoss, throttleLoss,
    bestSteerLoss, bestThrottleLoss, lr, message, startedAt,
    history, config, system, systemHistory, modelInfo, checkpoints,
    // computed
    status, progressPercent, currentGpu,
    // actions
    fetchInitialData, fetchHistory,
    startTraining, stopTraining, pauseTraining, resumeTraining,
    connectWs, dispose,
  }
})
