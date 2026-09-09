/**
 * SystemMonitor.vue — 系统资源监控面板
 *
 * 复刻 reference 右侧面板设计：
 * - GPU 利用率（渐变进度条：绿→黄→红）
 * - CPU 负载（蓝色进度条）
 * - 内存使用（蓝色进度条 + GB 显示）
 * - GPU 温度（温度计样式）
 * - 磁盘使用
 */
<template>
  <div class="sys-monitor-card">
    <h3 class="card-title">系统监控</h3>

    <div class="metrics">
      <!-- GPU 利用率 -->
      <div class="metric">
        <div class="metric-header">
          <span class="metric-label">GPU 利用率</span>
          <span class="metric-value" :class="gpuColorClass">{{ gpu.util_percent }}%</span>
        </div>
        <div class="progress-track">
          <div
            class="progress-fill progress-fill--gpu"
            :style="{ width: gpu.util_percent + '%' }"
          ></div>
        </div>
      </div>

      <!-- CPU 负载 -->
      <div class="metric">
        <div class="metric-header">
          <span class="metric-label">CPU 负载</span>
          <span class="metric-value">{{ sys.cpu_percent }}%</span>
        </div>
        <div class="progress-track">
          <div
            class="progress-fill progress-fill--cpu"
            :style="{ width: sys.cpu_percent + '%' }"
          ></div>
        </div>
      </div>

      <!-- 内存 (RAM) -->
      <div class="metric">
        <div class="metric-header">
          <span class="metric-label">内存 (RAM)</span>
          <span class="metric-value">{{ sys.memory_used_gb }}GB</span>
        </div>
        <div class="progress-track">
          <div
            class="progress-fill progress-fill--cpu"
            :style="{ width: sys.memory_percent + '%' }"
          ></div>
        </div>
      </div>

      <!-- GPU 温度 -->
      <div class="temp-row">
        <span class="material-symbols-outlined temp-icon">thermometer</span>
        <div class="temp-info">
          <div class="temp-label">核心温度</div>
          <div class="temp-value">
            {{ gpu.temp_c }}<span class="temp-unit">°C</span>
          </div>
        </div>
        <span class="temp-badge" :class="tempBadgeClass">{{ tempLabel }}</span>
      </div>

      <!-- 磁盘 -->
      <div class="disk-row">
        <span class="material-symbols-outlined disk-icon">database</span>
        <div class="progress-track disk-track">
          <div
            class="progress-fill progress-fill--disk"
            :style="{ width: sys.disk_percent + '%' }"
          ></div>
        </div>
        <span class="disk-label">{{ sys.disk_percent }}% 磁盘</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 系统监控面板
 * 展示 CPU/GPU/内存/温度/磁盘 实时数据
 */
import { computed, inject } from 'vue'
import { useTrainingStore } from '@/stores/training'

const store = useTrainingStore()

/** 系统快照 */
const sys = computed(() => store.system)

/** 当前 GPU */
const gpu = computed(() => store.currentGpu || {
  util_percent: 0,
  mem_used_mb: 0,
  mem_total_mb: 8188,
  mem_percent: 0,
  temp_c: 0,
  name: '',
  index: 0,
})

/** GPU 颜色类名 */
const gpuColorClass = computed(() => {
  const u = gpu.value.util_percent
  if (u > 85) return 'value-red'
  if (u > 60) return 'value-yellow'
  return 'value-green'
})

/** 温度标签 */
const tempLabel = computed(() => {
  const t = gpu.value.temp_c
  if (t > 80) return '过热'
  if (t > 65) return '偏高'
  return '正常'
})

/** 温度 badge 样式 */
const tempBadgeClass = computed(() => {
  const t = gpu.value.temp_c
  if (t > 80) return 'badge-hot'
  if (t > 65) return 'badge-warm'
  return 'badge-cool'
})
</script>

<style scoped>
.sys-monitor-card {
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
  margin-bottom: 1rem;
}

/* ---------- 指标 ---------- */
.metrics {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}

.metric-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.375rem;
}

.metric-label {
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.02em;
}

.metric-value {
  font-family: var(--font-mono);
  font-size: var(--font-size-mono);
  font-weight: 600;
  color: var(--color-primary);
}
.metric-value.value-green { color: var(--color-success); }
.metric-value.value-yellow { color: var(--color-secondary-container); }
.metric-value.value-red { color: var(--color-temp-red); }

/* ---------- 进度条 ---------- */
.progress-track {
  width: 100%;
  height: 0.75rem;
  background: var(--color-surface-container);
  border-radius: var(--radius-full);
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: var(--radius-full);
  transition: width 0.5s ease;
}

/* GPU 利用率：渐变绿→黄→红 */
.progress-fill--gpu {
  background: linear-gradient(90deg, #10B981, #fea619, #EF4444);
  background-size: 200% 100%;
  background-position: right center;
}

.progress-fill--cpu {
  background: var(--color-primary);
}

.progress-fill--disk {
  background: var(--color-text-muted);
}

/* ---------- 温度行 ---------- */
.temp-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.5rem 0.75rem;
  background: var(--color-surface-container-low);
  border-radius: var(--radius-md);
  border: 1px solid rgba(226, 232, 240, 0.5);
}

.temp-icon {
  color: var(--color-temp-red);
  font-size: 1.25rem;
}

.temp-info {
  flex: 1;
}

.temp-label {
  font-size: 0.625rem;
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
}

.temp-value {
  font-family: var(--font-mono);
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--color-on-surface);
}

.temp-unit {
  font-size: var(--font-size-body-sm);
  font-weight: 400;
}

.temp-badge {
  font-size: 0.625rem;
  font-weight: 700;
  padding: 0.125rem 0.5rem;
  border-radius: var(--radius-sm);
  text-transform: uppercase;
}

.badge-hot {
  color: var(--color-temp-red);
  background: rgba(239, 68, 68, 0.1);
}

.badge-warm {
  color: var(--color-secondary);
  background: rgba(254, 166, 25, 0.1);
}

.badge-cool {
  color: var(--color-success);
  background: rgba(16, 185, 129, 0.1);
}

/* ---------- 磁盘行 ---------- */
.disk-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.disk-icon {
  color: var(--color-text-muted);
}

.disk-track {
  flex: 1;
  height: 0.375rem;
}

.disk-label {
  font-family: var(--font-mono);
  font-size: 0.625rem;
  color: var(--color-text-muted);
  white-space: nowrap;
}
</style>
