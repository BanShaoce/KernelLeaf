/**
 * ModelInfo.vue — 模型信息卡片 + Checkpoint 列表
 *
 * 展示：
 * - KernelLeaf 共享骨干 + 双回归头架构信息
 * - 总参数量
 * - Checkpoint 文件列表（含下载链接）
 */
<template>
  <div class="model-info-card">
    <!-- 模型拓扑 -->
    <h3 class="card-title">模型拓扑</h3>

    <div class="model-list">
      <!-- SteerNet -->
      <div class="model-item model-item--active">
        <div class="model-item-header">
          <span class="model-name">转向回归头</span>
          <span class="model-badge model-badge--active">活跃</span>
        </div>
        <div class="model-desc">共享轻量 ResNet 特征 + Tanh 输出</div>
        <div class="model-params">{{ fmtParams(steerParams) }} 参数</div>
      </div>

      <!-- ThrottleNet -->
      <div class="model-item model-item--standby">
        <div class="model-item-header">
          <span class="model-name">油门回归头</span>
          <span class="model-badge model-badge--standby">活跃</span>
        </div>
        <div class="model-desc">共享轻量 ResNet 特征 + Sigmoid 输出</div>
        <div class="model-params">{{ fmtParams(throttleParams) }} 参数</div>
      </div>
    </div>

    <!-- 总参数量 -->
    <div class="total-params">
      <span class="total-label">总参数量</span>
      <span class="total-value">{{ fmtParams(totalParams) }}</span>
    </div>

    <!-- Checkpoint 列表 -->
    <div class="checkpoints-section">
      <h4 class="checkpoints-title">Checkpoints</h4>
      <ul class="checkpoints-list">
        <li v-for="ckpt in checkpoints" :key="ckpt.name" class="checkpoint-item">
          <span class="material-symbols-outlined checkpoint-icon">description</span>
          <div class="checkpoint-info">
            <span class="checkpoint-name">{{ ckpt.name }}</span>
            <span class="checkpoint-size">{{ ckpt.size_mb }} MB</span>
          </div>
          <a v-if="ckpt.download_url" :href="ckpt.download_url" class="checkpoint-download" title="下载">
            <span class="material-symbols-outlined">download</span>
          </a>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 模型信息卡片组件
 */
import { computed } from 'vue'
import { useTrainingStore } from '@/stores/training'

const store = useTrainingStore()

const steerParams = computed(() => store.modelInfo?.steer_model?.parameters ?? 0)
const throttleParams = computed(() => store.modelInfo?.throttle_model?.parameters ?? 0)
const totalParams = computed(() => store.modelInfo?.total_parameters ?? 0)
const checkpoints = computed(() => store.checkpoints)

/** 格式化参数量（如 11.18M） */
function fmtParams(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K'
  return String(n)
}
</script>

<style scoped>
.model-info-card {
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

/* ---------- 模型列表 ---------- */
.model-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.model-item {
  padding: 0.75rem;
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
}

.model-item--active {
  background: rgba(0, 74, 198, 0.05);
  border-color: rgba(0, 74, 198, 0.2);
}

.model-item--standby {
  background: var(--color-surface-container-low);
}

.model-item-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.25rem;
}

.model-name {
  font-size: var(--font-size-body-sm);
  font-weight: 700;
  color: var(--color-primary);
}

.model-item--standby .model-name {
  color: var(--color-on-surface);
}

.model-badge {
  font-size: 0.625rem;
  font-weight: 700;
  padding: 0.125rem 0.375rem;
  border-radius: var(--radius-sm);
}

.model-badge--active {
  background: var(--color-primary);
  color: var(--color-on-primary);
}

.model-badge--standby {
  background: var(--color-outline-variant);
  color: var(--color-on-surface-variant);
}

.model-desc {
  font-size: 0.6875rem;
  color: var(--color-text-muted);
}

.model-params {
  font-family: var(--font-mono);
  font-size: 0.6875rem;
  color: var(--color-text-muted);
  margin-top: 0.125rem;
}

/* ---------- 总参数量 ---------- */
.total-params {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 0.75rem;
  margin-top: 0.75rem;
  border-top: 1px dashed var(--color-border);
}

.total-label {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.total-value {
  font-family: var(--font-mono);
  font-size: var(--font-size-body-sm);
  font-weight: 700;
}

/* ---------- Checkpoints ---------- */
.checkpoints-section {
  margin-top: 1rem;
  padding-top: 0.75rem;
  border-top: 1px solid var(--color-border);
}

.checkpoints-title {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.5rem;
}

.checkpoints-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
  max-height: 180px;
  overflow-y: auto;
}

.checkpoint-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.375rem 0.5rem;
  border-radius: var(--radius-sm);
  transition: background 0.15s;
}
.checkpoint-item:hover {
  background: var(--color-surface-container-low);
}

.checkpoint-icon {
  font-size: 1rem;
  color: var(--color-text-muted);
}

.checkpoint-info {
  flex: 1;
  min-width: 0;
}

.checkpoint-name {
  display: block;
  font-family: var(--font-mono);
  font-size: 0.6875rem;
  color: var(--color-text-main);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.checkpoint-size {
  font-size: 0.625rem;
  color: var(--color-text-muted);
}

.checkpoint-download {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 1.75rem;
  height: 1.75rem;
  border-radius: var(--radius-sm);
  color: var(--color-primary);
  text-decoration: none;
  transition: background 0.15s;
  flex-shrink: 0;
}
.checkpoint-download:hover {
  background: rgba(0, 74, 198, 0.1);
}
.checkpoint-download .material-symbols-outlined {
  font-size: 1rem;
}
</style>
