/**
 * TrainingHistory.vue — 训练历史表格
 *
 * 列：epoch | steer_loss | throttle_loss | best_steer | best_throttle | epoch_time | n_batches
 * - 最佳 epoch 行高亮（绿色背景 + 左侧绿条）
 * - 支持按列排序（点击表头）
 */
<template>
  <div class="history-card">
    <!-- 头部 -->
    <div class="history-header">
      <h3 class="history-title">训练日志 (实时)</h3>
      <div class="history-actions">
        <button class="action-btn" @click="exportCSV">导出 CSV</button>
      </div>
    </div>

    <!-- 表格 -->
    <div class="table-wrapper">
      <table class="history-table">
        <thead>
          <tr>
            <th
              v-for="col in columns"
              :key="col.key"
              :class="['th-cell', { 'sortable': col.sortable, 'sorted': sortKey === col.key }]"
              @click="col.sortable && toggleSort(col.key)"
            >
              {{ col.label }}
              <span v-if="sortKey === col.key" class="sort-icon">
                {{ sortAsc ? '▲' : '▼' }}
              </span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in sortedHistory"
            :key="row.epoch"
            :class="['tr-row', { 'tr-row--best': row.is_best_steer || row.is_best_throttle }]"
          >
            <td class="td-cell td-epoch">
              <span class="epoch-num">{{ String(row.epoch).padStart(3, '0') }}</span>
              <span v-if="row.is_best_steer || row.is_best_throttle" class="best-tag">最佳</span>
            </td>
            <td class="td-cell mono">{{ row.steer_loss.toFixed(6) }}</td>
            <td class="td-cell mono">{{ row.throttle_loss.toFixed(6) }}</td>
            <td class="td-cell mono">{{ row.best_steer_loss.toFixed(6) }}</td>
            <td class="td-cell mono">{{ row.best_throttle_loss.toFixed(6) }}</td>
            <td class="td-cell mono text-muted">{{ row.epoch_time.toFixed(2) }}s</td>
            <td class="td-cell mono text-muted">{{ row.lr.toExponential(2) }}</td>
            <td class="td-cell mono text-muted">{{ row.n_batches }}</td>
          </tr>
          <!-- 空状态 -->
          <tr v-if="!store.history.length">
            <td :colspan="columns.length" class="empty-cell">
              暂无训练数据，请在左侧配置参数并点击“开始”
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 训练历史表格组件
 * 支持按 loss 列排序，最佳 epoch 绿色高亮
 */
import { ref, computed } from 'vue'
import { useTrainingStore, type TrainHistoryEntry } from '@/stores/training'

const store = useTrainingStore()

/** 列定义 */
const columns = [
  { key: 'epoch', label: '轮数', sortable: false },
  { key: 'steer_loss', label: '转向损失', sortable: true },
  { key: 'throttle_loss', label: '油门损失', sortable: true },
  { key: 'best_steer_loss', label: '最佳转向', sortable: true },
  { key: 'best_throttle_loss', label: '最佳油门', sortable: true },
  { key: 'epoch_time', label: '单步耗时', sortable: true },
  { key: 'lr', label: '学习率', sortable: true },
  { key: 'n_batches', label: '批次数', sortable: false },
]

/** 排序状态 */
const sortKey = ref<string>('')
const sortAsc = ref(true)

function toggleSort(key: string) {
  if (sortKey.value === key) {
    sortAsc.value = !sortAsc.value
  } else {
    sortKey.value = key
    sortAsc.value = true
  }
}

/** 排序后的历史 */
const sortedHistory = computed(() => {
  const list = [...store.history]
  if (!sortKey.value) return list.reverse() // 默认倒序（最新在前）

  return list.sort((a, b) => {
    const aVal = (a as any)[sortKey.value]
    const bVal = (b as any)[sortKey.value]
    const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0
    return sortAsc.value ? cmp : -cmp
  })
})

/** 导出 CSV */
function exportCSV() {
  const headers = columns.map(c => c.label).join(',')
  const rows = store.history.map(row =>
    [row.epoch, row.steer_loss, row.throttle_loss, row.best_steer_loss,
     row.best_throttle_loss, row.epoch_time, row.lr, row.n_batches].join(','),
  )
  const csv = [headers, ...rows].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `training_history_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.history-card {
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  overflow: hidden;
  box-shadow: var(--shadow-card);
}

.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem var(--spacing-card);
  border-bottom: 1px solid var(--color-border);
}

.history-title {
  font-size: var(--font-size-title);
  font-weight: 600;
  color: var(--color-text-main);
}

.history-actions {
  display: flex;
  gap: 0.5rem;
}

.action-btn {
  padding: 0.375rem 0.75rem;
  font-size: var(--font-size-label);
  font-weight: 700;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: transparent;
  color: var(--color-text-main);
  cursor: pointer;
  transition: background 0.15s;
}
.action-btn:hover {
  background: var(--color-surface-container-low);
}

/* ---------- 表格 ---------- */
.table-wrapper {
  overflow-x: auto;
}

.history-table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}

.th-cell {
  padding: 0.75rem 1.5rem;
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  background: var(--color-surface-container-low);
  white-space: nowrap;
  user-select: none;
}

.th-cell.sortable {
  cursor: pointer;
  transition: color 0.15s;
}
.th-cell.sortable:hover {
  color: var(--color-primary);
}
.th-cell.sorted {
  color: var(--color-primary);
}

.sort-icon {
  font-size: 0.625rem;
  margin-left: 0.25rem;
}

/* ---------- 行 ---------- */
.tr-row {
  border-bottom: 1px solid var(--color-border);
  transition: background 0.15s;
}
.tr-row:hover {
  background: var(--color-surface-container-low);
}

/* 最佳 epoch 高亮 */
.tr-row--best {
  background: rgba(16, 185, 129, 0.05);
  border-left: 4px solid var(--color-success);
}
.tr-row--best:hover {
  background: rgba(16, 185, 129, 0.08);
}

/* ---------- 单元格 ---------- */
.td-cell {
  padding: 1rem 1.5rem;
  font-size: var(--font-size-body-sm);
  color: var(--color-text-main);
}
.td-cell.mono {
  font-family: var(--font-mono);
  font-size: var(--font-size-mono);
}
.td-cell.text-muted {
  color: var(--color-text-muted);
}

.td-epoch {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.epoch-num {
  font-family: var(--font-mono);
  font-size: var(--font-size-body-sm);
  font-weight: 700;
  color: var(--color-success);
}

.tr-row--best .epoch-num {
  color: var(--color-success);
}

.best-tag {
  font-size: 0.625rem;
  font-weight: 700;
  padding: 0.125rem 0.375rem;
  border-radius: var(--radius-sm);
  background: rgba(16, 185, 129, 0.2);
  color: var(--color-success);
}

/* ---------- 空状态 ---------- */
.empty-cell {
  padding: 3rem;
  text-align: center;
  color: var(--color-text-muted);
  font-size: var(--font-size-body-sm);
}
</style>
