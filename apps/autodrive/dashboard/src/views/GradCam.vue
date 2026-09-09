/**
 * GradCam.vue — Tab 2: Grad-CAM 可视化
 *
 * 布局：
 * - 顶部参数面板（checkpoint 选择、样本数、目标输出、透明度、种子）
 * - 响应式热力图网格（3 列 → 2 列 → 1 列）
 * - 支持悬停放大 + 从历史记录加载
 */
<template>
  <div class="gradcam-page">
    <!-- 顶部导航 -->
    <AppHeader />

    <!-- 主内容 -->
    <main class="main-content">
      <!-- 页面标题 -->
      <div class="page-header">
        <div>
          <h1 class="page-title">Grad-CAM 可视化</h1>
          <p class="page-desc">解释深度神经网络在自动驾驶视觉任务中的关注点。</p>
        </div>
        <button class="btn-history" @click="loadFromHistory">
          <span class="material-symbols-outlined">history</span>
          从历史记录加载
        </button>
      </div>

      <!-- 参数面板 -->
      <section class="param-panel">
        <div class="param-row">
          <!-- Steer Checkpoint -->
          <div class="param-group param-group--flex">
            <label class="param-label">转向检查点</label>
            <div class="select-wrapper">
              <select v-model="params.steer_ckpt" class="param-select">
                <option v-for="ckpt in steerCheckpoints" :key="ckpt.name" :value="ckpt.name">
                  {{ ckpt.display_name || ckpt.name }}
                </option>
              </select>
              <span class="material-symbols-outlined select-arrow">expand_more</span>
            </div>
          </div>
          <!-- Throttle Checkpoint -->
          <div class="param-group param-group--flex">
            <label class="param-label">油门检查点</label>
            <div class="select-wrapper">
              <select v-model="params.throttle_ckpt" class="param-select">
                <option v-for="ckpt in throttleCheckpoints" :key="ckpt.name" :value="ckpt.name">
                  {{ ckpt.display_name || ckpt.name }}
                </option>
              </select>
              <span class="material-symbols-outlined select-arrow">expand_more</span>
            </div>
          </div>
          <!-- 样本数 -->
          <div class="param-group param-group--narrow">
            <label class="param-label">样本数</label>
            <input v-model.number="params.sample_count" type="number" class="param-input" min="1" max="24" />
          </div>
          <!-- 目标输出 (segmented button) -->
          <div class="param-group">
            <label class="param-label">目标输出</label>
            <div class="segmented">
              <button
                v-for="opt in targetOptions"
                :key="opt.value"
                :class="['seg-btn', { 'seg-btn--active': params.target === opt.value }]"
                @click="params.target = opt.value"
              >
                {{ opt.label }}
              </button>
            </div>
          </div>
          <!-- Alpha 透明度 -->
          <div class="param-group param-group--narrow">
            <label class="param-label">透明度 ({{ params.alpha }})</label>
            <input v-model.number="params.alpha" type="range" min="0" max="1" step="0.05" class="param-range" />
          </div>
          <!-- 随机种子 -->
          <div class="param-group param-group--narrow">
            <label class="param-label">随机种子</label>
            <input v-model.number="params.seed" type="number" class="param-input" />
          </div>
          <!-- 生成按钮 -->
          <button class="btn-generate" disabled title="请使用命令行生成 Grad-CAM">
            <span v-if="loading" class="material-symbols-outlined spin">sync</span>
            <span v-else class="material-symbols-outlined">analytics</span>
            命令行生成
          </button>
        </div>
      </section>

      <!-- 热力图网格 -->
      <section class="heatmap-grid">
        <!-- 空状态 -->
        <div v-if="records.length === 0 && !loading" class="empty-state">
          <span class="material-symbols-outlined empty-icon">image_search</span>
          <p>请使用命令行生成 Grad-CAM 可视化结果</p>
          <p class="empty-hint">或点击"从历史记录加载"查看之前的结果</p>
        </div>

        <!-- 加载状态 -->
        <div v-if="loading" class="loading-state">
          <span class="material-symbols-outlined spin">sync</span>
          <p>正在生成热力图...</p>
        </div>

        <!-- 热力图卡片 -->
        <div
          v-for="(rec, idx) in records"
          :key="idx"
          class="heatmap-card"
          @mousemove="onCardMouseMove($event, idx)"
          @mouseleave="onCardMouseLeave(idx)"
        >
          <!-- 图片容器 -->
          <div class="heatmap-image" ref="imgContainers">
            <img
              v-if="rec.image_url"
              :src="rec.image_url"
              :alt="`Grad-CAM ${rec.target} #${rec.index}`"
              class="gradcam-img"
            />
            <div v-else class="gradcam-placeholder" :class="`target-${rec.target}`">
              <div class="gradcam-overlay"></div>
            </div>
          </div>
          <!-- 底部信息 -->
          <div class="card-footer">
            <span class="sample-label">
              样本 #{{ String(rec.index).padStart(3, '0') }}
              <span class="target-badge">{{ rec.target === 'steer' ? '转向' : '油门' }}</span>
            </span>
            <div class="card-actions">
              <span class="material-symbols-outlined action-icon" title="查看">visibility</span>
              <span class="material-symbols-outlined action-icon" title="下载">download</span>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
/**
 * Grad-CAM 可视化页面
 * 调用 /api/viz/gradcam 生成热力图，展示在响应式网格中
 */
import { ref, computed, onMounted, inject } from 'vue'
import AppHeader from '@/components/AppHeader.vue'
import { vizApi, modelApi } from '@/api'
import { useTrainingStore } from '@/stores/training'

const store = useTrainingStore()
const showToast = inject<any>('showToast', () => {})

/** 参数 */
const params = ref<{
  steer_ckpt: string
  throttle_ckpt: string
  sample_count: number
  target: string
  alpha: number
  seed: number
}>({
  steer_ckpt: '',
  throttle_ckpt: '',
  sample_count: 6,
  target: 'both',
  alpha: 0.45,
  seed: 256,
})

/** 目标选项 */
const targetOptions = [
  { value: 'both', label: '全部' },
  { value: 'steer', label: '转向' },
  { value: 'throttle', label: '油门' },
]

/** 加载状态 */
const loading = ref(false)

/** 热力图记录 */
const records = ref<any[]>([])

/** Checkpoint 列表 */
const steerCheckpoints = computed(() =>
  store.checkpoints.filter((c: any) => c.name.toLowerCase().endsWith('.npz')),
)
const throttleCheckpoints = computed(() =>
  store.checkpoints.filter((c: any) => c.name.toLowerCase().endsWith('.npz')),
)

/** 图片容器引用（用于计算鼠标位置） */
const imgContainers = ref<HTMLElement[]>([])

/** 从历史记录加载 */
async function loadFromHistory() {
  loading.value = true
  try {
    const result = await vizApi.getList()
    if (result.manifest && result.manifest.length > 0) {
      records.value = result.manifest
      showToast(`已加载 ${result.manifest.length} 条历史记录`, 'info')
    } else {
      showToast('暂无历史记录', 'warning')
    }
  } catch (err: any) {
    showToast('加载失败: ' + err.message, 'error')
  } finally {
    loading.value = false
  }
}

/** 鼠标移入热力图 — 变换焦点 */
function onCardMouseMove(e: MouseEvent, idx: number) {
  const el = e.currentTarget as HTMLElement
  const rect = el.getBoundingClientRect()
  const x = ((e.clientX - rect.left) / rect.width) * 100
  const y = ((e.clientY - rect.top) / rect.height) * 100
  const img = el.querySelector('.gradcam-placeholder') as HTMLElement
  if (img) {
    img.style.transformOrigin = `${x}% ${y}%`
  }
}

function onCardMouseLeave(idx: number) {
  // reset handled by CSS :hover
}

/** 初始化时加载 checkpoints */
onMounted(async () => {
  try {
    const data = await modelApi.getCheckpoints()
    store.checkpoints = data.checkpoints || []
  } catch { /* ignore */ }
})
</script>

<style scoped>
/* ---------- 整体布局 ---------- */
.gradcam-page {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.main-content {
  flex: 1;
  overflow-y: auto;
  padding: var(--spacing-gutter);
}

/* ---------- 页面标题 ---------- */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: var(--spacing-stack);
  padding-top: 0.5rem;
}

.page-title {
  font-size: var(--font-size-display-lg);
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--color-text-main);
}

.page-desc {
  font-size: var(--font-size-body-sm);
  color: var(--color-text-muted);
  margin-top: 0.25rem;
}

.btn-history {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  color: var(--color-text-main);
  font-size: var(--font-size-label);
  font-weight: 700;
  cursor: pointer;
  transition: background 0.15s;
  white-space: nowrap;
}
.btn-history:hover {
  background: var(--color-surface-container);
}

/* ---------- 参数面板 ---------- */
.param-panel {
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  padding: var(--spacing-card);
  margin-bottom: var(--spacing-stack);
  box-shadow: var(--shadow-card);
}

.param-row {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 1rem;
}

.param-group {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.param-group--flex {
  flex: 1;
  min-width: 180px;
}
.param-group--narrow {
  width: 100px;
}

.param-label {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  white-space: nowrap;
}

.param-input,
.param-select {
  padding: 0.5rem 0.75rem;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-family: var(--font-mono);
  font-size: var(--font-size-body-sm);
  color: var(--color-text-main);
  outline: none;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.param-input:focus,
.param-select:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(0, 74, 198, 0.15);
}

.param-select {
  width: 100%;
  appearance: none;
  padding-right: 2rem;
}

.select-wrapper {
  position: relative;
}

.select-arrow {
  position: absolute;
  right: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  font-size: 1.25rem;
  color: var(--color-text-muted);
  pointer-events: none;
}

.param-range {
  width: 100%;
  accent-color: var(--color-primary);
}

/* ---------- 分段按钮 (Segmented) ---------- */
.segmented {
  display: flex;
  padding: 2px;
  background: var(--color-surface-container-low);
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
}

.seg-btn {
  flex: 1;
  padding: 0.375rem 0.5rem;
  border: none;
  background: transparent;
  font-size: 0.6875rem;
  font-weight: 700;
  color: var(--color-text-muted);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}
.seg-btn--active {
  background: var(--color-surface-container-lowest);
  color: var(--color-primary);
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

/* ---------- 生成按钮 ---------- */
.btn-generate {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.625rem 1.5rem;
  background: var(--color-primary);
  color: var(--color-on-primary);
  border: none;
  border-radius: var(--radius-md);
  font-size: var(--font-size-label);
  font-weight: 700;
  cursor: pointer;
  transition: opacity 0.15s, transform 0.1s;
  white-space: nowrap;
  margin-left: auto;
}
.btn-generate:active:not(:disabled) {
  transform: scale(0.98);
}
.btn-generate:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* ---------- 热力图网格 ---------- */
.heatmap-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-gutter);
}

@media (max-width: 1100px) {
  .heatmap-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 640px) {
  .heatmap-grid { grid-template-columns: 1fr; }
}

/* ---------- 热力图卡片 ---------- */
.heatmap-card {
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(8px);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  overflow: hidden;
  box-shadow: var(--shadow-card);
  transition: box-shadow 0.2s;
}
.heatmap-card:hover {
  box-shadow: var(--shadow-hover);
}

.heatmap-image {
  position: relative;
  aspect-ratio: 16 / 9;
  overflow: hidden;
  cursor: crosshair;
}

/* ---------- 真实 Grad-CAM 图片 ---------- */
.gradcam-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: var(--radius-md);
}

/* ---------- 占位 Grad-CAM 图 ---------- */
.gradcam-placeholder {
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  position: relative;
  display: flex;
  align-items: flex-end;
  transition: transform 0.5s ease;
}
.heatmap-card:hover .gradcam-placeholder {
  transform: scale(1.05);
}

/* 无图片记录的视觉占位层 */
.gradcam-overlay {
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse at 30% 50%, rgba(255,0,0,0.6) 0%, rgba(255,165,0,0.3) 40%, transparent 70%),
              radial-gradient(ellipse at 65% 55%, rgba(0,100,255,0.4) 0%, transparent 60%);
  pointer-events: none;
}
.target-throttle .gradcam-overlay {
  background: radial-gradient(ellipse at 50% 40%, rgba(0,200,255,0.5) 0%, rgba(0,100,200,0.3) 40%, transparent 70%),
              radial-gradient(ellipse at 55% 60%, rgba(255,200,0,0.3) 0%, transparent 60%);
}

/* 底部标签栏 */
.gradcam-labels {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 0.75rem;
  background: linear-gradient(to top, rgba(0,0,0,0.6), transparent);
  display: flex;
  gap: 0.5rem;
  z-index: 1;
}

.label-tag,
.pred-tag {
  padding: 0.25rem 0.5rem;
  border-radius: var(--radius-sm);
  font-size: 0.625rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #fff;
  backdrop-filter: blur(4px);
}

.label-tag {
  background: rgba(255,255,255,0.2);
  border: 1px solid rgba(255,255,255,0.3);
}

.pred-tag {
  background: rgba(37, 99, 235, 0.4);
  border: 1px solid rgba(37, 99, 235, 0.5);
}
.pred-tag.pred-error {
  background: rgba(239, 68, 68, 0.4);
  border-color: rgba(239, 68, 68, 0.5);
}

/* 状态圆点 */
.status-dot-img {
  position: absolute;
  top: 0.75rem;
  right: 0.75rem;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  z-index: 1;
}
.dot-ok {
  background: var(--color-success);
  box-shadow: 0 0 8px rgba(16, 185, 129, 0.8);
}
.dot-error {
  background: var(--color-temp-red);
  box-shadow: 0 0 8px rgba(239, 68, 68, 0.8);
}

/* 悬停放大提示 */
.zoom-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0,0,0,0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  opacity: 0;
  transition: opacity 0.2s;
  z-index: 2;
  pointer-events: none;
}
.heatmap-card:hover .zoom-overlay {
  opacity: 1;
}
.zoom-overlay .material-symbols-outlined {
  font-size: 1.5rem;
  color: #fff;
}
.zoom-overlay {
  color: #fff;
  font-size: var(--font-size-body-sm);
  font-weight: 700;
}

/* ---------- 卡片底部 ---------- */
.card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem;
}

.sample-label {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.target-badge {
  font-size: 0.625rem;
  padding: 0.125rem 0.375rem;
  border-radius: var(--radius-sm);
  background: var(--color-primary-fixed-dim);
  color: var(--color-primary);
  font-weight: 700;
}

.card-actions {
  display: flex;
  gap: 0.25rem;
}

.action-icon {
  font-size: 1.125rem;
  color: var(--color-text-muted);
  cursor: pointer;
  transition: color 0.15s;
}
.action-icon:hover {
  color: var(--color-primary);
}

/* ---------- 空状态 ---------- */
.empty-state {
  grid-column: 1 / -1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  color: var(--color-text-muted);
  text-align: center;
}

.empty-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
  opacity: 0.4;
}

.empty-hint {
  font-size: var(--font-size-body-sm);
  margin-top: 0.5rem;
  opacity: 0.6;
}

/* ---------- 加载状态 ---------- */
.loading-state {
  grid-column: 1 / -1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  color: var(--color-primary);
}

/* ---------- 旋转动画 ---------- */
.spin {
  animation: spin 1s linear infinite;
}
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>
