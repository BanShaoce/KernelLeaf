/**
 * DataManagement.vue — Tab 4: 数据管理
 *
 * 布局：
 * - 左侧：当前数据集信息（训练集 / 验证集数量 + 比例条）
 * - 右侧：重新划分数据参数
 * - 底部：数据集文件列表
 */
<template>
  <div class="data-page">
    <!-- 顶部导航 -->
    <AppHeader />

    <!-- 主内容 -->
    <main class="main-content">
      <!-- 页面标题 -->
      <div class="page-header">
        <div>
          <h1 class="page-title">当前数据集信息</h1>
          <p class="page-desc">查看和管理用于自动驾驶模型训练的数据集划分。</p>
        </div>
        <div class="header-actions">
          <button class="btn-outline" disabled>
            <span class="material-symbols-outlined">upload_file</span>
            导入数据
          </button>
          <button class="btn-primary" disabled>
            <span class="material-symbols-outlined">play_circle</span>
            命令行处理
          </button>
        </div>
      </div>

      <!-- 面板网格 (12 列) -->
      <div class="panel-grid">
        <!-- ===== 当前数据集信息 (4 列) ===== -->
        <section class="left-section">
          <div class="card">
            <div class="card-header-row">
              <h3 class="card-title">当前数据集</h3>
              <span class="api-badge">GET /api/data/info</span>
            </div>

            <div class="data-list">
              <!-- 训练集 -->
              <div class="data-item">
                <div class="data-icon data-icon--train">
                  <span class="material-symbols-outlined">folder_shared</span>
                </div>
                <div class="data-info">
                  <p class="data-name">训练集 (Train)</p>
                  <p class="data-file">train.txt</p>
                </div>
                <div class="data-count">
                  <p class="data-number">{{ dataInfo?.train?.count?.toLocaleString() ?? '--' }}</p>
                  <p class="data-unit">Images</p>
                </div>
              </div>

              <!-- 验证集 -->
              <div class="data-item">
                <div class="data-icon data-icon--val">
                  <span class="material-symbols-outlined">verified</span>
                </div>
                <div class="data-info">
                  <p class="data-name">验证集 (Val)</p>
                  <p class="data-file">val.txt</p>
                </div>
                <div class="data-count">
                  <p class="data-number">{{ dataInfo?.val?.count?.toLocaleString() ?? '--' }}</p>
                  <p class="data-unit">Images</p>
                </div>
              </div>
            </div>

            <!-- 数据集分布比例 -->
            <div class="split-section">
              <div class="split-header">
                <span class="split-label">数据集分布 (Split Ratio)</span>
                <span class="split-value">{{ trainRatio }}% : {{ valRatio }}%</span>
              </div>
              <div class="split-bar">
                <div class="split-train" :style="{ width: trainRatio + '%' }"></div>
                <div class="split-val" :style="{ width: valRatio + '%' }"></div>
              </div>
            </div>
          </div>
        </section>

        <!-- ===== 重新划分数据 (8 列) ===== -->
        <section class="right-section">
          <div class="card">
            <div class="card-header-row">
              <div>
                <h3 class="card-title">重新划分数据</h3>
                <p class="card-desc">当前为只读视图；请使用命令行生成 manifest。</p>
              </div>
              <span class="api-badge">CLI ONLY</span>
            </div>

            <form class="regenerate-form" @submit.prevent>
              <!-- 图像目录 -->
              <div class="form-group form-group--full">
                <label class="form-label">图像目录 (image_dir)</label>
                <div class="input-with-icon">
                  <span class="material-symbols-outlined input-icon">folder_open</span>
                  <input v-model="regenerateParams.image_dir" type="text" class="form-input form-input--icon" disabled />
                </div>
              </div>

              <!-- 训练集比例 (range slider) -->
              <div class="form-group">
                <label class="form-label">训练集比例 (train_ratio)</label>
                <div class="range-row">
                  <input
                    v-model.number="regenerateParams.train_ratio"
                    type="range"
                    class="range-slider"
                    min="0.5"
                    max="0.95"
                    step="0.05"
                    disabled
                  />
                  <span class="range-value">{{ regenerateParams.train_ratio.toFixed(2) }}</span>
                </div>
              </div>

              <!-- 随机种子 -->
              <div class="form-group">
                <label class="form-label">随机种子 (seed)</label>
                <div class="input-with-icon">
                  <input v-model.number="regenerateParams.seed" type="number" class="form-input form-input--icon" disabled />
                  <button type="button" class="icon-btn-inside" disabled>
                    <span class="material-symbols-outlined">autorenew</span>
                  </button>
                </div>
              </div>

              <!-- 提交按钮 -->
              <div class="form-actions">
                <button type="submit" class="btn-submit" disabled>
                  <span v-if="regenerating" class="material-symbols-outlined spin">sync</span>
                  <span v-else class="material-symbols-outlined">shuffle</span>
                  命令行重新划分
                </button>
              </div>
            </form>

            <!-- 结果提示 -->
            <div v-if="regenerateResult" class="result-banner result-success">
              <span class="material-symbols-outlined">check_circle</span>
              {{ regenerateResult.message }}
              <span class="result-detail">
                训练集: {{ regenerateResult.train }} | 验证集: {{ regenerateResult.val }}
              </span>
            </div>
          </div>
        </section>

        <!-- ===== 文件列表 (12 列全宽) ===== -->
        <section class="full-section">
          <div class="card">
            <div class="card-header-row">
              <h3 class="card-title">数据集文件列表</h3>
              <div class="table-tools">
                <div class="search-box">
                  <span class="material-symbols-outlined search-icon">search</span>
                  <input v-model="searchQuery" type="text" class="search-input" placeholder="搜索文件名..." />
                </div>
                <button class="btn-icon-only">
                  <span class="material-symbols-outlined">filter_list</span>
                </button>
              </div>
            </div>

            <!-- 文件表格 -->
            <div class="table-wrapper">
              <table class="file-table">
                <thead>
                  <tr>
                    <th>文件名</th>
                    <th>格式</th>
                    <th>分辨率</th>
                    <th>所属集合</th>
                    <th>最后修改</th>
                    <th class="text-right">操作</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="file in filteredFiles" :key="file.path" class="file-row">
                    <td class="file-name" :title="file.path">{{ file.name }}</td>
                    <td class="text-muted">JPEG</td>
                    <td class="text-muted">
                      steer:{{ file.steer?.toFixed(3) }} thr:{{ file.throttle?.toFixed(3) }}
                    </td>
                    <td>
                      <span :class="['set-tag', file.set === 'Train' ? 'set-train' : 'set-val']">
                        {{ file.set }}
                      </span>
                    </td>
                    <td class="text-muted">{{ file.size_bytes ? (file.size_bytes / 1024).toFixed(0) + ' KB' : '-' }}</td>
                    <td class="text-right">
                      <a :href="file.preview_url" target="_blank" class="link-btn">预览</a>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- 分页 -->
            <div class="pagination">
              <span class="page-info">显示 1-{{ filteredFiles.length }}，共 {{ realFiles.length }} 个文件</span>
              <div class="page-btns">
                <button class="page-btn" disabled>上一页</button>
                <button class="page-btn page-btn--active">1</button>
                <button class="page-btn" disabled>2</button>
                <button class="page-btn" disabled>下一页</button>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
/**
 * 数据管理页面
 * GET /api/data/info 获取数据集信息
 * V11 看板只读取数据摘要，重新划分由命令行完成
 */
import { ref, computed, onMounted } from 'vue'
import AppHeader from '@/components/AppHeader.vue'
import { dataApi } from '@/api'

/** 数据集信息 */
const dataInfo = ref<any>(null)

/** 训练集比例 */
const trainRatio = computed(() => {
  if (!dataInfo.value) return 0
  const total = dataInfo.value.train.count + dataInfo.value.val.count
  if (!total) return 0
  return Math.round((dataInfo.value.train.count / total) * 100)
})
const valRatio = computed(() => dataInfo.value ? Math.max(0, 100 - trainRatio.value) : 0)

/** 重新划分参数 */
const regenerateParams = ref({
  image_dir: './data_generated_track_thro_2',
  train_ratio: 0.8,
  seed: 256,
})

/** 重新划分状态 */
const regenerating = ref(false)
const regenerateResult = ref<any>(null)

/** 搜索 */
const searchQuery = ref('')

/** V11 默认不暴露源文件路径。 */
const realFiles = ref<any[]>([])
const loadingFiles = ref(false)

/** 获取文件列表 */
async function fetchFileList() {
  loadingFiles.value = true
  try {
    const res = await fetch('/api/data/files?limit=200')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    console.log('[Data] files loaded:', data.total, 'total,', data.files?.length, 'in page')
    realFiles.value = data.files || []
  } catch (err: any) {
    console.error('[Data] fetchFileList failed:', err.message)
    realFiles.value = []
  } finally {
    loadingFiles.value = false
  }
}

const filteredFiles = computed(() => {
  if (!searchQuery.value) return realFiles.value
  const q = searchQuery.value.toLowerCase()
  return realFiles.value.filter((f: any) => f.name.toLowerCase().includes(q))
})

/** 初始化 */
onMounted(async () => {
  try { dataInfo.value = await dataApi.getInfo() } catch { /* */ }
  fetchFileList()
})
</script>

<style scoped>
/* ---------- 整体布局 ---------- */
.data-page {
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

.header-actions {
  display: flex;
  gap: 0.75rem;
}

.btn-outline {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: var(--font-size-body-sm);
  font-weight: 500;
  color: var(--color-text-main);
  cursor: pointer;
  transition: background 0.15s;
}
.btn-outline:hover { background: var(--color-surface-container-low); }

.btn-primary {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--color-primary);
  color: var(--color-on-primary);
  border: none;
  border-radius: var(--radius-md);
  font-size: var(--font-size-body-sm);
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.15s;
}

/* ---------- 面板网格 ---------- */
.panel-grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: var(--spacing-gutter);
}

.left-section { grid-column: span 4; }
.right-section { grid-column: span 8; }
.full-section { grid-column: span 12; }

@media (max-width: 1024px) {
  .panel-grid { grid-template-columns: 1fr; }
  .left-section, .right-section, .full-section { grid-column: span 1; }
}

/* ---------- 通用卡片 ---------- */
.card {
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  padding: var(--spacing-card);
  box-shadow: var(--shadow-card);
  height: 100%;
}

.card-title {
  font-size: var(--font-size-title);
  font-weight: 600;
  color: var(--color-text-main);
}

.card-desc {
  font-size: var(--font-size-body-sm);
  color: var(--color-text-muted);
}

.card-header-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 1.5rem;
}

.api-badge {
  font-family: var(--font-mono);
  font-size: 0.625rem;
  color: var(--color-text-muted);
  white-space: nowrap;
}

/* ---------- 数据集信息 ---------- */
.data-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.data-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: var(--color-surface-container-low);
  border-radius: var(--radius-md);
}

.data-icon {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.data-icon--train {
  background: rgba(0, 74, 198, 0.1);
  color: var(--color-primary);
}

.data-icon--val {
  background: rgba(254, 166, 25, 0.1);
  color: var(--color-secondary);
}

.data-info { flex: 1; }
.data-name { font-size: var(--font-size-body-sm); font-weight: 500; color: var(--color-text-main); }
.data-file { font-size: 0.75rem; color: var(--color-text-muted); }

.data-count { text-align: right; }
.data-number { font-size: var(--font-size-title); font-weight: 700; color: var(--color-text-main); }
.data-unit { font-size: 0.625rem; color: var(--color-text-muted); text-transform: uppercase; }

/* ---------- 比例条 ---------- */
.split-section {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--color-border);
}

.split-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 0.5rem;
}
.split-label { font-size: var(--font-size-label); font-weight: 700; color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.split-value { font-family: var(--font-mono); font-size: var(--font-size-body-sm); color: var(--color-primary); }

.split-bar {
  height: 0.5rem;
  background: var(--color-surface-container-high);
  border-radius: var(--radius-full);
  overflow: hidden;
  display: flex;
}
.split-train { background: var(--color-primary); }
.split-val { background: var(--color-secondary-container); }

/* ---------- 重新划分表单 ---------- */
.regenerate-form {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

.form-group { display: flex; flex-direction: column; gap: 0.5rem; }
.form-group--full { grid-column: span 2; }

.form-label {
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-main);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.form-input {
  padding: 0.625rem 0.75rem;
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-family: var(--font-mono);
  font-size: var(--font-size-body-sm);
  color: var(--color-text-main);
  outline: none;
  transition: border-color 0.15s;
}
.form-input:focus {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 2px rgba(0, 74, 198, 0.15);
}
.form-input--icon { padding-left: 2.5rem; }

.input-with-icon { position: relative; }
.input-icon {
  position: absolute;
  left: 0.75rem;
  top: 50%;
  transform: translateY(-50%);
  font-size: 1.25rem;
  color: var(--color-text-muted);
}

.icon-btn-inside {
  position: absolute;
  right: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  color: var(--color-text-muted);
  cursor: pointer;
  padding: 0.25rem;
}
.icon-btn-inside:hover { color: var(--color-primary); }

/* ---------- Range Slider ---------- */
.range-row {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.range-slider {
  flex: 1;
  height: 0.375rem;
  accent-color: var(--color-primary);
}

.range-value {
  font-family: var(--font-mono);
  font-size: var(--font-size-body-sm);
  font-weight: 700;
  color: var(--color-primary);
  min-width: 3rem;
  text-align: center;
}

/* ---------- 提交按钮 ---------- */
.form-actions {
  grid-column: span 2;
  display: flex;
  justify-content: flex-end;
  padding-top: 0.5rem;
}

.btn-submit {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 2rem;
  background: var(--color-primary);
  color: var(--color-on-primary);
  border: none;
  border-radius: var(--radius-md);
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 74, 198, 0.2);
  transition: opacity 0.15s, transform 0.1s;
}
.btn-submit:active:not(:disabled) { transform: scale(0.98); }
.btn-submit:disabled { opacity: 0.6; cursor: not-allowed; }

/* ---------- 结果提示 ---------- */
.result-banner {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 1rem;
  padding: 0.75rem 1rem;
  border-radius: var(--radius-md);
  font-size: var(--font-size-body-sm);
  font-weight: 500;
}
.result-success {
  background: rgba(16, 185, 129, 0.08);
  color: var(--color-success);
  border: 1px solid rgba(16, 185, 129, 0.2);
}
.result-detail {
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: var(--font-size-mono);
  opacity: 0.8;
}

/* ---------- 文件表格 ---------- */
.table-tools {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.search-box {
  position: relative;
}
.search-icon {
  position: absolute;
  left: 0.5rem;
  top: 50%;
  transform: translateY(-50%);
  font-size: 1.125rem;
  color: var(--color-text-muted);
}
.search-input {
  padding: 0.375rem 0.75rem 0.375rem 2rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: var(--font-size-body-sm);
  outline: none;
}
.search-input:focus { border-color: var(--color-primary); }

.btn-icon-only {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: transparent;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-icon-only:hover { background: var(--color-surface-container-low); }

.table-wrapper { overflow-x: auto; }

.file-table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}
.file-table thead th {
  padding: 0.75rem 1rem;
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  background: rgba(242, 244, 246, 0.5);
  border-bottom: 1px solid var(--color-border);
}

.file-row {
  border-bottom: 1px solid var(--color-border);
  transition: background 0.15s;
}
.file-row:hover { background: rgba(242, 244, 246, 0.3); }

.file-row td {
  padding: 1rem;
  font-size: var(--font-size-body-sm);
}
.file-name { font-family: var(--font-mono); color: var(--color-text-main); }
.text-muted { color: var(--color-text-muted); }
.text-right { text-align: right; }

.set-tag {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  border-radius: var(--radius-sm);
  font-size: 0.625rem;
  font-weight: 700;
  text-transform: uppercase;
}
.set-train {
  background: rgba(0, 74, 198, 0.1);
  color: var(--color-primary);
}
.set-val {
  background: rgba(254, 166, 25, 0.1);
  color: var(--color-secondary);
}

.link-btn {
  background: none;
  border: none;
  color: var(--color-primary);
  font-weight: 500;
  cursor: pointer;
}
.link-btn:hover { text-decoration: underline; }

/* ---------- 分页 ---------- */
.pagination {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-top: 1px solid var(--color-border);
  margin-top: 0.5rem;
}

.page-info {
  font-size: var(--font-size-body-sm);
  color: var(--color-text-muted);
}

.page-btns { display: flex; gap: 0.5rem; }
.page-btn {
  padding: 0.375rem 0.75rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: transparent;
  font-size: var(--font-size-body-sm);
  cursor: pointer;
  transition: background 0.15s;
}
.page-btn:hover:not(:disabled) { background: var(--color-surface-container-low); }
.page-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.page-btn--active { background: var(--color-primary); color: #fff; border-color: var(--color-primary); }

.spin { animation: spin 1s linear infinite; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
</style>
