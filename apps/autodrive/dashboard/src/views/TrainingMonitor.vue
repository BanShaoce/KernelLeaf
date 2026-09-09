/**
 * TrainingMonitor.vue — Tab 1: 训练监控（主页面）
 *
 * 三栏布局：
 * - 左侧 280px：参数配置 + 操作按钮 + 进度条
 * - 中央 flex:1：Loss 曲线 + System 曲线 + 训练历史表格
 * - 右侧 300px：系统监控 + 模型信息
 */
<template>
  <div class="training-page">
    <!-- 顶部导航 -->
    <AppHeader />

    <!-- 主内容区：三栏布局 -->
    <div class="main-content">
      <!-- ========== 左侧面板 (280px) ========== -->
      <aside class="left-panel">
        <!-- 参数配置 -->
        <div class="card">
          <div class="card-header-row">
            <h3 class="card-title">训练参数</h3>
            <!-- 状态指示灯 -->
            <div class="status-indicator">
              <span
                class="status-dot"
                :class="{
                  'status-dot--running': store.status === 'running',
                  'status-dot--paused': store.status === 'paused',
                }"
              ></span>
              <span class="status-text">{{ statusLabel }}</span>
            </div>
          </div>

          <!-- 参数表单 -->
          <div class="param-form">
            <div class="form-group">
              <label class="form-label">Manifest</label>
              <input
                v-model="localConfig.manifest"
                type="text"
                class="form-input"
                :disabled="store.running"
                placeholder="data/DonkeyCar/manifest.jsonl"
              />
            </div>
            <div class="form-group">
              <label class="form-label">地图（一次训练一个）</label>
              <input
                v-model="localConfig.map"
                type="text"
                class="form-input"
                :disabled="store.running"
                placeholder="warren-track"
                required
              />
            </div>
            <!-- batch_size -->
            <div class="form-group">
              <label class="form-label">批次大小 (BATCH SIZE)</label>
              <input
                v-model.number="localConfig.batch_size"
                type="number"
                class="form-input"
                :disabled="store.running"
                placeholder="96"
              />
            </div>
            <!-- epochs -->
            <div class="form-group">
              <label class="form-label">轮数 (EPOCHS)</label>
              <input
                v-model.number="localConfig.epochs"
                type="number"
                class="form-input"
                :disabled="store.running"
                placeholder="100"
              />
            </div>
            <!-- lr + weight_decay -->
            <div class="form-row">
              <div class="form-group">
                <label class="form-label">学习率 (LR)</label>
                <input
                  v-model.number="localConfig.lr"
                  type="text"
                  class="form-input"
                  :disabled="store.running"
                  placeholder="1e-4"
                />
              </div>
              <div class="form-group">
                <label class="form-label">权重衰减</label>
                <input
                  v-model.number="localConfig.weight_decay"
                  type="text"
                  class="form-input"
                  :disabled="store.running"
                  placeholder="1e-4"
                />
              </div>
            </div>
            <!-- grad_clip_norm + device -->
            <div class="form-row">
              <div class="form-group">
                <label class="form-label">梯度裁剪</label>
                <input
                  v-model.number="localConfig.grad_clip_norm"
                  type="text"
                  class="form-input"
                  :disabled="store.running"
                  placeholder="5.0"
                />
              </div>
              <div class="form-group">
                <label class="form-label">设备</label>
                <select v-model="localConfig.device" class="form-input form-select" :disabled="store.running">
                  <option value="cuda">cuda:0</option>
                  <option value="cpu">cpu</option>
                </select>
              </div>
            </div>
          </div>

          <!-- 训练进度条 -->
          <div class="progress-section">
            <div class="progress-header">
              <span class="progress-label">训练进度</span>
              <span class="progress-text">{{ store.epoch }} / {{ store.totalEpochs }} 轮</span>
            </div>
            <div class="progress-track">
              <div
                class="progress-fill"
                :style="{ width: store.progressPercent + '%' }"
              ></div>
            </div>
          </div>

          <!-- 操作按钮组 -->
          <div class="action-buttons">
            <!-- 开始／恢复：idle 或 paused 时显示 -->
            <button
              v-if="store.status === 'idle' || store.status === 'paused'"
              class="btn btn-start"
              @click="handleStart"
            >
              <span class="material-symbols-outlined">play_arrow</span>
              {{ store.status === 'paused' ? '恢复' : '开始' }}
            </button>
            <!-- 暂停：仅 running 时可用 -->
            <button
              class="btn btn-pause"
              :disabled="store.status !== 'running'"
              @click="handlePause"
            >
              <span class="material-symbols-outlined">pause</span>
              暂停
            </button>
            <!-- 停止：非 idle 时可用 -->
            <button
              class="btn btn-stop"
              :disabled="store.status === 'idle'"
              @click="handleStop"
            >
              <span class="material-symbols-outlined">stop</span>
              停止
            </button>
          </div>
          <p class="process-message">{{ store.message }}</p>

          <!-- 当前 Loss 信息 -->
          <div v-if="store.status !== 'idle'" class="current-loss">
            <div class="loss-row">
              <span class="loss-label">转向 Loss</span>
              <span class="loss-value loss-steer">{{ fmtLoss(store.steerLoss) }}</span>
            </div>
            <div class="loss-row">
              <span class="loss-label">油门 Loss</span>
              <span class="loss-value loss-throttle">{{ fmtLoss(store.throttleLoss) }}</span>
            </div>
          </div>
        </div>
      </aside>

      <!-- ========== 中央内容区 (flex:1) ========== -->
      <section class="center-panel">
        <!-- Loss 曲线 -->
        <div class="chart-wrap"><LossChart /></div>

        <!-- 系统资源曲线 -->
        <div class="chart-wrap"><SystemChart /></div>

        <!-- 训练历史表格 -->
        <div class="chart-wrap"><TrainingHistory /></div>
      </section>

      <!-- ========== 右侧面板 (300px) ========== -->
      <aside class="right-panel">
        <!-- 系统监控 -->
        <SystemMonitor />

        <!-- 模型信息 + Checkpoints -->
        <ModelInfo />
      </aside>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 训练监控主页面
 * 初始化时加载所有数据，建立 WebSocket，管理训练生命周期
 */
import { ref, computed, onMounted, onUnmounted, inject } from 'vue'
import AppHeader from '@/components/AppHeader.vue'
import LossChart from '@/components/LossChart.vue'
import SystemChart from '@/components/SystemChart.vue'
import TrainingHistory from '@/components/TrainingHistory.vue'
import SystemMonitor from '@/components/SystemMonitor.vue'
import ModelInfo from '@/components/ModelInfo.vue'
import { useTrainingStore } from '@/stores/training'

const store = useTrainingStore()
const showToast = inject<any>('showToast', () => {})

/** 本地配置（双向绑定表单） */
const localConfig = ref({
  manifest: 'data/DonkeyCar/manifest.jsonl',
  map: '',
  batch_size: 32,
  epochs: 10,
  lr: 1e-3,
  weight_decay: 1e-4,
  grad_clip_norm: 5.0,
  device: 'cpu',
})

/** 状态标签 */
const statusLabel = computed(() => {
  switch (store.status) {
    case 'running': return '运行中'
    case 'paused': return '已暂停'
    default: return '空闲'
  }
})

/** 格式化 Loss */
function fmtLoss(v: number | null): string {
  if (v === null || v === undefined) return '--'
  return v.toFixed(6)
}

async function handleStart() {
  try {
    if (store.status === 'paused') {
      await store.resumeTraining()
      showToast('训练已恢复', 'success')
    } else {
      if (!localConfig.value.map.trim()) {
        throw new Error('请先指定一个地图')
      }
      await store.startTraining(localConfig.value)
      showToast('KernelLeaf 训练已启动', 'success')
    }
  } catch (err: any) {
    showToast(err.message || '启动失败', 'error')
  }
}

async function handlePause() {
  try {
    await store.pauseTraining()
    showToast('训练已暂停', 'warning')
  } catch (err: any) {
    showToast(err.message || '暂停失败', 'error')
  }
}

async function handleStop() {
  try {
    await store.stopTraining()
    showToast('停止信号已发送', 'info')
  } catch (err: any) {
    showToast(err.message || '停止失败', 'error')
  }
}

/** 初始化：加载数据 + 建立 WebSocket */
onMounted(async () => {
  try {
    await store.fetchInitialData()
    localConfig.value = { ...store.config }
    store.connectWs()
  } catch (err: any) {
    showToast('数据加载失败: ' + err.message, 'error')
  }
})

onUnmounted(() => {
  store.dispose()
})
</script>

<style scoped>
/* ---------- 整体布局 ---------- */
.training-page {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.main-content {
  display: flex;
  flex: 1;
  padding: var(--spacing-gutter);
  gap: var(--spacing-gutter);
  min-height: 0;
}

/* ---------- 左侧面板 ---------- */
.left-panel {
  width: var(--sidebar-width);
  flex-shrink: 0;
  overflow-y: auto;
}

/* ---------- 中央区域 ---------- */
.center-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-stack);
  overflow-y: auto;
  min-width: 0;
}

/* 防止 flex 压缩 — 确保内容溢出时出现滚动条 */
.chart-wrap {
  flex-shrink: 0;
}

/* ---------- 右侧面板 ---------- */
.right-panel {
  width: var(--aside-width);
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-stack);
  overflow-y: auto;
}

/* ---------- 通用卡片 ---------- */
.card {
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  padding: var(--spacing-card);
  box-shadow: var(--shadow-card);
}

.card-title {
  font-size: var(--font-size-title);
  font-weight: 600;
  color: var(--color-text-main);
}

.card-header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
}

/* ---------- 状态指示器 ---------- */
.status-indicator {
  display: flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.25rem 0.5rem;
  border-radius: var(--radius-full);
  background: rgba(16, 185, 129, 0.1);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--color-text-muted);
}
.status-dot--running {
  background: var(--color-success);
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
.status-dot--paused {
  background: var(--color-secondary-container);
}

.status-text {
  font-size: 0.625rem;
  font-weight: 700;
  text-transform: uppercase;
}
.status-dot--running ~ .status-text { color: var(--color-success); }
.status-dot--paused ~ .status-text { color: var(--color-secondary); }

/* ---------- 表单 ---------- */
.param-form {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
}

.form-label {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.form-input {
  width: 100%;
  padding: 0.5rem 0.75rem;
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-family: var(--font-mono);
  font-size: var(--font-size-mono);
  color: var(--color-text-main);
  outline: none;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.form-input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(0, 74, 198, 0.15);
}

.form-select {
  appearance: none;
  cursor: pointer;
}

/* ---------- 进度条 ---------- */
.progress-section {
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--color-border);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.progress-label {
  font-size: var(--font-size-body-sm);
  font-weight: 500;
  color: var(--color-text-muted);
}

.progress-text {
  font-family: var(--font-mono);
  font-size: var(--font-size-mono);
  color: var(--color-primary);
}

.progress-track {
  width: 100%;
  height: 0.5rem;
  background: var(--color-surface-container);
  border-radius: var(--radius-full);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--color-primary);
  border-radius: var(--radius-full);
  transition: width 1s ease;
}

/* ---------- 操作按钮 ---------- */
.action-buttons {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-top: 1.5rem;
}

.btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.625rem;
  border: none;
  border-radius: var(--radius-md);
  font-size: var(--font-size-body-sm);
  font-weight: 700;
  cursor: pointer;
  transition: opacity 0.15s, transform 0.1s;
}
.btn:active:not(:disabled) {
  transform: scale(0.98);
}
.btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.btn-start {
  background: var(--color-success);
  color: #fff;
}
.btn-pause {
  background: var(--color-secondary-container);
  color: var(--color-on-secondary-container);
}
.btn-stop {
  background: var(--color-temp-red);
  color: #fff;
}

.process-message {
  margin-top: 0.75rem;
  color: var(--color-text-muted);
  font-size: var(--font-size-body-sm);
  line-height: 1.4;
  overflow-wrap: anywhere;
}

/* ---------- 当前 Loss ---------- */
.current-loss {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--color-border);
}

.loss-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.25rem 0;
}

.loss-label {
  font-size: var(--font-size-body-sm);
  color: var(--color-text-muted);
}

.loss-value {
  font-family: var(--font-mono);
  font-size: var(--font-size-mono);
  font-weight: 600;
}

.loss-steer { color: var(--color-secondary-fixed-dim); }
.loss-throttle { color: var(--color-primary); }

/* ---------- 响应式 ---------- */
@media (max-width: 1200px) {
  .right-panel {
    display: none;
  }
}

@media (max-width: 768px) {
  .main-content {
    flex-direction: column;
    padding: 0.75rem;
    gap: 0.75rem;
  }
  .left-panel {
    width: 100%;
  }
  .right-panel {
    display: flex;
    width: 100%;
  }
}
</style>
