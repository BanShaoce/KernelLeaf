/**
 * Validation.vue — Tab 3: 验证评估
 *
 * 布局：
 * - 左侧：验证配置参数（checkpoint 选择、batch_size、device）
 * - 右侧：结果卡片（Steer MSE、Throttle MSE、耗时）
 * - 底部：系统资源条
 */
<template>
  <div class="validation-page">
    <!-- 顶部导航 -->
    <AppHeader />

    <!-- 主内容 -->
    <main class="main-content">
      <!-- 页面标题 -->
      <div class="page-header">
        <div>
          <h2 class="page-title">验证评估</h2>
          <p class="page-desc">查看 KernelLeaf 训练日志保存的最新验证指标</p>
        </div>
        <div class="header-right">
          <button class="btn-new-exp" @click="handleEvaluate">
            <span class="material-symbols-outlined">refresh</span>
            加载最新结果
          </button>
          <div class="status-badge">
            <span class="status-dot" :class="evaluating ? 'pulsing-dot' : ''"></span>
            <span class="status-text">{{ evaluating ? '评估中...' : '系统就绪' }}</span>
          </div>
        </div>
      </div>

      <!-- 面板网格 (12 列) -->
      <div class="panel-grid">
        <!-- ===== 左侧：配置面板 (4 列) ===== -->
        <section class="left-section">
          <!-- 验证配置 -->
          <div class="card">
            <h3 class="card-title">
              <span class="material-symbols-outlined title-icon">settings_input_component</span>
              验证配置参数
            </h3>

            <div class="config-form">
              <!-- Steer Checkpoint -->
              <div class="form-group">
                <label class="form-label">转向检查点 (STEER CKPT)</label>
                <select v-model="config.steer_ckpt" class="form-select">
                  <option v-for="ckpt in steerCheckpoints" :key="ckpt.name" :value="ckpt.name">
                    {{ ckpt.display_name || ckpt.name }}
                  </option>
                </select>
              </div>
              <!-- Throttle Checkpoint -->
              <div class="form-group">
                <label class="form-label">油门检查点 (THROTTLE CKPT)</label>
                <select v-model="config.throttle_ckpt" class="form-select">
                  <option v-for="ckpt in throttleCheckpoints" :key="ckpt.name" :value="ckpt.name">
                    {{ ckpt.display_name || ckpt.name }}
                  </option>
                </select>
              </div>
              <!-- batch_size + device -->
              <div class="form-row">
                <div class="form-group">
                  <label class="form-label">批次大小</label>
                  <input v-model.number="config.batch_size" type="number" class="form-input" />
                </div>
                <div class="form-group">
                  <label class="form-label">设备</label>
                  <select v-model="config.device" class="form-select">
                    <option value="auto">训练时使用的设备</option>
                    <option value="cpu">CPU</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- 开始评估按钮 -->
            <button class="btn-eval" :disabled="evaluating" @click="handleEvaluate">
              <span v-if="evaluating" class="material-symbols-outlined spin">sync</span>
              <span v-else class="material-symbols-outlined">play_arrow</span>
              {{ evaluating ? '加载中...' : '加载最新验证结果' }}
            </button>
          </div>

          <!-- 任务详情 -->
          <div class="card">
            <h3 class="card-label-caps">任务详情</h3>
            <div class="detail-list">
              <div class="detail-row">
                <span class="detail-label">样本数量</span>
                <span class="detail-value">{{ result?.n_samples?.toLocaleString() || '--' }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">单样本延迟</span>
                <span class="detail-value">{{ result?.time_per_sample_ms != null ? result.time_per_sample_ms.toFixed(2) + 'ms' : '--' }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">吞吐量</span>
                <span class="detail-value throughput">
                  {{ throughput }}
                </span>
              </div>
            </div>
          </div>
        </section>

        <!-- ===== 右侧：结果区域 (8 列) ===== -->
        <section class="right-section">
          <!-- 指标卡片 -->
          <div class="metrics-row">
            <!-- Steer MSE -->
            <div class="metric-card">
              <div class="metric-bg-icon">
                <span class="material-symbols-outlined">near_me</span>
              </div>
              <p class="metric-label-caps">转向均方误差 (MSE)</p>
              <div class="metric-value-large">{{ result ? result.steer_mse.toFixed(6) : '--' }}</div>
              <div class="metric-bar">
                <div class="metric-bar-fill metric-bar--steer" :style="{ width: steerBarWidth }"></div>
              </div>
              <p class="metric-trend trend-down">
                <span class="material-symbols-outlined">trending_down</span>
                训练日志中的最新结果
              </p>
            </div>

            <!-- Throttle MSE -->
            <div class="metric-card">
              <div class="metric-bg-icon">
                <span class="material-symbols-outlined">speed</span>
              </div>
              <p class="metric-label-caps">油门均方误差 (MSE)</p>
              <div class="metric-value-large">{{ result ? result.throttle_mse.toFixed(6) : '--' }}</div>
              <div class="metric-bar">
                <div class="metric-bar-fill metric-bar--throttle" :style="{ width: throttleBarWidth }"></div>
              </div>
              <p class="metric-trend trend-ok">
                <span class="material-symbols-outlined">check_circle</span>
                训练日志中的最新结果
              </p>
            </div>

            <!-- 耗时 -->
            <div class="metric-card">
              <div class="metric-bg-icon">
                <span class="material-symbols-outlined">timer</span>
              </div>
              <p class="metric-label-caps">评估耗时</p>
              <div class="metric-value-large">
                {{ result ? result.time_sec.toFixed(2) : '--' }}<span class="metric-unit">s</span>
              </div>
              <div class="metric-bar">
                <div class="metric-bar-fill metric-bar--time" :style="{ width: timeBarWidth }"></div>
              </div>
              <p class="metric-trend trend-muted">最近一轮的训练与验证耗时</p>
            </div>
          </div>

          <!-- 误差分布图 (ECharts 散点 + 回归线) -->
          <div class="card chart-card-placeholder">
            <div class="card-header-row">
              <h3 class="card-title">误差分布分析</h3>
              <div class="card-actions">
                <button class="action-btn-sm">导出 CSV</button>
                <button class="action-btn-sm">快照</button>
              </div>
            </div>
            <div ref="scatterChartRef" class="scatter-chart-body"></div>
          </div>
        </section>
      </div>

      <!-- 底部资源监控条 -->
      <footer class="resource-strip">
        <div class="resource-item">
          <div class="resource-header">
            <span class="resource-label">GPU 利用率</span>
            <span class="resource-value">{{ gpu.util_percent }}%</span>
          </div>
          <div class="resource-track">
            <div class="resource-fill resource-fill--gpu" :style="{ width: gpu.util_percent + '%' }"></div>
          </div>
        </div>
        <div class="resource-item">
          <div class="resource-header">
            <span class="resource-label">显存占用</span>
            <span class="resource-value">{{ gpu.mem_used_mb }} / {{ gpu.mem_total_mb }} MB</span>
          </div>
          <div class="resource-track">
            <div class="resource-fill resource-fill--vram" :style="{ width: gpu.mem_percent + '%' }"></div>
          </div>
        </div>
        <div class="resource-item">
          <div class="resource-header">
            <span class="resource-label">CPU 核心负载</span>
            <span class="resource-value">{{ sys.cpu_percent }}%</span>
          </div>
          <div class="resource-track">
            <div class="resource-fill resource-fill--cpu" :style="{ width: sys.cpu_percent + '%' }"></div>
          </div>
        </div>
        <div class="resource-item">
          <div class="resource-header">
            <span class="resource-label">存储 I/O</span>
            <span class="resource-value">{{ sys.disk_percent }}%</span>
          </div>
          <div class="resource-track">
            <div class="resource-fill resource-fill--disk" :style="{ width: sys.disk_percent + '%' }"></div>
          </div>
        </div>
      </footer>
    </main>
  </div>
</template>

<script setup lang="ts">
/**
 * 验证评估页面
 * GET /api/eval/validation 读取训练期间保存的验证结果
 */
import { ref, computed, onMounted, onUnmounted, inject } from 'vue'
import * as echarts from 'echarts'
import AppHeader from '@/components/AppHeader.vue'
import { evalApi, modelApi } from '@/api'
import { useTrainingStore } from '@/stores/training'

const store = useTrainingStore()
const showToast = inject<any>('showToast', () => {})

/** 配置 */
const config = ref({
  steer_ckpt: '',
  throttle_ckpt: '',
  batch_size: 96,
  device: 'auto',
})

/** 评估状态 */
const evaluating = ref(false)
const result = ref<any>(null)

/** 系统状态（底部资源条） */
const sys = computed(() => store.system)
const gpu = computed(() => store.currentGpu || { util_percent: 0, mem_used_mb: 0, mem_total_mb: 8188, mem_percent: 0, temp_c: 0 })

/** Checkpoint 列表 */
const steerCheckpoints = computed(() =>
  store.checkpoints.filter((c: any) => c.name.toLowerCase().endsWith('.npz')),
)
const throttleCheckpoints = computed(() =>
  store.checkpoints.filter((c: any) => c.name.toLowerCase().endsWith('.npz')),
)

/** 吞吐量 */
const throughput = computed(() => {
  if (!result.value) return '--'
  if (!result.value.time_sec) return '--'
  return (result.value.n_samples / result.value.time_sec).toFixed(1) + ' FPS'
})

/** 进度条宽度（美化） */
const steerBarWidth = computed(() => result.value ? Math.min((1 - result.value.steer_mse / 0.02) * 100, 100) + '%' : '0%')
const throttleBarWidth = computed(() => result.value ? Math.min((1 - result.value.throttle_mse / 0.005) * 100, 100) + '%' : '0%')
const timeBarWidth = computed(() => result.value ? Math.min((result.value.time_sec / 1) * 50, 100) + '%' : '0%')

// ============================================================================
// ECharts 散点图 + 回归分析
// ============================================================================
const scatterChartRef = ref<HTMLElement | null>(null)
let scatterChart: echarts.ECharts | null = null

/** 仅展示后端提供的真实逐样本数据；V11 默认不记录时保持空图。 */
function generateScatterData() {
  const normalPoints: number[][] = result.value?.scatter_points || []
  const anomalyPoints: number[][] = result.value?.anomaly_points || []
  return { normalPoints, anomalyPoints, regLine: [], slope: 0, intercept: 0 }
}

function initScatterChart() {
  if (!scatterChartRef.value) return
  scatterChart = echarts.init(scatterChartRef.value)

  const { normalPoints, anomalyPoints, regLine, slope, intercept } = generateScatterData()

  scatterChart.setOption({
    grid: { top: 20, right: 30, bottom: 40, left: 55, containLabel: true },
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        const data = params.data || []
        const seriesName = params.seriesName || ''
        const tag = seriesName.includes('异常') ? ' ⚠️ 异常' : ''
        return `<b>${seriesName}</b>${tag}<br/>真实值: <b>${data[0]?.toFixed(4)}</b><br/>预测值: <b>${data[1]?.toFixed(4)}</b><br/>误差: <b>${(data[1] - data[0]).toFixed(4)}</b>`
      },
    },
    xAxis: {
      type: 'value',
      name: '真实值 (Ground Truth)',
      nameLocation: 'middle',
      nameGap: 28,
      nameTextStyle: { color: '#64748B', fontSize: 11 },
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: { color: '#64748B', fontSize: 10 },
      splitLine: { lineStyle: { color: '#F1F5F9' } },
    },
    yAxis: {
      type: 'value',
      name: '预测值 (Prediction)',
      nameLocation: 'middle',
      nameGap: 40,
      nameTextStyle: { color: '#64748B', fontSize: 11 },
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: { color: '#64748B', fontSize: 10 },
      splitLine: { lineStyle: { color: '#F1F5F9' } },
    },
    series: [
      {
        name: '正常样本',
        type: 'scatter',
        data: normalPoints,
        symbolSize: 6,
        itemStyle: {
          color: 'rgba(37, 99, 235, 0.45)',
          borderColor: 'rgba(37, 99, 235, 0.8)',
          borderWidth: 1,
        },
        emphasis: {
          itemStyle: { color: '#2563eb' },
          scale: 1.8,
        },
      },
      {
        name: '异常样本',
        type: 'scatter',
        data: anomalyPoints,
        symbolSize: 10,
        itemStyle: {
          color: 'rgba(239, 68, 68, 0.7)',
          borderColor: '#EF4444',
          borderWidth: 2,
        },
        emphasis: {
          itemStyle: { color: '#EF4444' },
          scale: 2,
        },
      },
      {
        name: '回归线',
        type: 'line',
        data: regLine,
        smooth: false,
        symbol: 'none',
        lineStyle: {
          color: '#10B981',
          width: 2,
          type: 'dashed',
        },
        markLine: {
          silent: true,
          symbol: 'none',
          label: { show: false },
          data: [], // no extra mark lines
        },
      },
    ],
  })
}

function onResize() { scatterChart?.resize() }

/** 执行评估 */
async function handleEvaluate() {
  evaluating.value = true
  result.value = null
  try {
    const cfg = {
      steer_checkpoint: config.value.steer_ckpt,
      throttle_checkpoint: config.value.throttle_ckpt,
      batch_size: config.value.batch_size,
      device: config.value.device,
    }
    result.value = await evalApi.validate(cfg)
    // 若未来日志包含逐样本预测，可直接渲染真实散点。
    if (scatterChart) {
      const { normalPoints, anomalyPoints, regLine } = generateScatterData()
      scatterChart.setOption({
        series: [
          { data: normalPoints },
          { data: anomalyPoints },
          { data: regLine },
        ],
      })
    }
    showToast('评估完成', 'success')
  } catch (err: any) {
    showToast('评估失败: ' + err.message, 'error')
  } finally {
    evaluating.value = false
  }
}

/** 初始化 */
onMounted(async () => {
  initScatterChart()
  window.addEventListener('resize', onResize)
  try {
    const data = await modelApi.getCheckpoints()
    store.checkpoints = data.checkpoints || []
  } catch { /* ignore */ }
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  scatterChart?.dispose()
})
</script>

<style scoped>
/* ---------- 整体布局 ---------- */
.validation-page {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.main-content {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-gutter);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-gutter);
}

/* ---------- 页面标题 ---------- */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
}

.page-title {
  font-size: var(--font-size-headline);
  font-weight: 600;
  color: var(--color-text-main);
}

.page-desc {
  font-size: var(--font-size-body-sm);
  color: var(--color-text-muted);
  margin-top: 0.25rem;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.btn-new-exp {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--color-primary);
  color: var(--color-on-primary);
  border: none;
  border-radius: var(--radius-md);
  font-weight: 700;
  cursor: pointer;
  transition: opacity 0.15s;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--color-success);
}
.status-text {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-main);
}

/* ---------- 面板网格 ---------- */
.panel-grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: var(--spacing-gutter);
}

.left-section {
  grid-column: span 4;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-stack);
}

.right-section {
  grid-column: span 8;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-stack);
}

@media (max-width: 1024px) {
  .panel-grid { grid-template-columns: 1fr; }
  .left-section, .right-section { grid-column: span 1; }
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
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.title-icon { color: var(--color-primary); }

.card-label-caps {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  margin-bottom: 1rem;
}

/* ---------- 配置表单 ---------- */
.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  margin-bottom: 1rem;
}

.form-label {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.form-input, .form-select {
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: var(--font-size-body-sm);
  color: var(--color-text-main);
  background: var(--color-surface);
  outline: none;
  transition: border-color 0.15s;
}
.form-input:focus, .form-select:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(0, 74, 198, 0.15);
}
.form-select { appearance: none; }

.form-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
}

/* ---------- 评估按钮 ---------- */
.btn-eval {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 1rem;
  background: var(--color-primary);
  color: var(--color-on-primary);
  border: none;
  border-radius: var(--radius-xl);
  font-size: var(--font-size-title);
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 74, 198, 0.2);
  transition: opacity 0.15s, transform 0.1s;
  margin-top: 1rem;
}
.btn-eval:active:not(:disabled) { transform: scale(0.98); }
.btn-eval:disabled { opacity: 0.6; cursor: not-allowed; }

/* ---------- 任务详情 ---------- */
.detail-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--color-border);
}
.detail-row:last-child { border-bottom: none; padding-bottom: 0; }

.detail-label { font-size: var(--font-size-body-sm); color: var(--color-text-muted); }
.detail-value { font-family: var(--font-mono); font-size: var(--font-size-body-md); font-weight: 700; }
.detail-value.throughput { color: var(--color-success); }

/* ---------- 指标卡片 ---------- */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-gutter);
}
@media (max-width: 768px) { .metrics-row { grid-template-columns: 1fr; } }

.metric-card {
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  padding: var(--spacing-card);
  box-shadow: var(--shadow-card);
  position: relative;
  overflow: hidden;
}

.metric-bg-icon {
  position: absolute;
  top: 0;
  right: 0;
  padding: 1rem;
  opacity: 0.1;
  font-size: 2.5rem;
  color: var(--color-primary);
}

.metric-label-caps {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}

.metric-value-large {
  font-size: var(--font-size-display-lg);
  font-weight: 700;
  color: var(--color-text-main);
  letter-spacing: -0.02em;
}

.metric-unit { font-size: var(--font-size-title); font-weight: 400; }

.metric-bar {
  width: 100%;
  height: 4px;
  background: var(--color-surface-container-low);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 1rem;
}
.metric-bar-fill { height: 100%; border-radius: 2px; }
.metric-bar--steer { background: var(--color-primary); }
.metric-bar--throttle { background: var(--color-secondary-container); }
.metric-bar--time { background: var(--color-gpu-purple); width: 45% !important; }

.metric-trend {
  font-size: 0.625rem;
  margin-top: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}
.trend-down { color: var(--color-success); }
.trend-ok { color: var(--color-text-muted); }
.trend-muted { color: var(--color-text-muted); }
.metric-trend .material-symbols-outlined { font-size: 0.75rem; }

/* ---------- 误差分布图 ---------- */
.chart-card-placeholder {
  flex: 1;
  min-height: 400px;
  display: flex;
  flex-direction: column;
}
.card-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  flex-shrink: 0;
}
.action-btn-sm {
  padding: 0.25rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: transparent;
  font-size: var(--font-size-label);
  font-weight: 700;
  cursor: pointer;
  transition: background 0.15s;
}
.action-btn-sm:hover { background: var(--color-surface-container-low); }

.scatter-chart-body {
  flex: 1;
  width: 100%;
  min-height: 300px;
}

/* ---------- 底部资源条 ---------- */
.resource-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: var(--spacing-gutter);
  padding-top: var(--spacing-stack);
  border-top: 1px solid var(--color-border);
}
@media (max-width: 768px) { .resource-strip { grid-template-columns: repeat(2, 1fr); } }

.resource-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.resource-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.resource-label { font-size: var(--font-size-label); font-weight: 700; color: var(--color-text-muted); text-transform: uppercase; }
.resource-value { font-family: var(--font-mono); font-size: var(--font-size-mono); }

.resource-track {
  width: 100%;
  height: 0.375rem;
  background: var(--color-surface-container-low);
  border-radius: 2px;
  overflow: hidden;
}
.resource-fill { height: 100%; border-radius: 2px; }
.resource-fill--gpu { background: var(--color-primary); }
.resource-fill--vram { background: var(--color-vram-orange); }
.resource-fill--cpu { background: var(--color-success); }
.resource-fill--disk { background: var(--color-gpu-purple); }

.spin { animation: spin 1s linear infinite; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
</style>
