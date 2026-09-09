/**
 * LossChart.vue — 训练损失收敛曲线（ECharts）
 *
 * 复刻 reference/autodrive_dashboard_1/code.html 设计：
 * - 双 Y 轴折线图 | steer=橙 #ffb95f | throttle=蓝 #2563eb
 * - 最佳 loss 用红色 pin 标记
 * - 训练未开始时显示演示数据，训练中逐 epoch 动态追加数据点
 */
<template>
  <div class="chart-card">
    <div class="chart-header">
      <h3 class="chart-title">训练损失收敛</h3>
      <div class="chart-legend">
        <div class="legend-item">
          <span class="legend-dot legend-dot--train"></span>
          <span class="legend-label">训练 / 验证</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot legend-dot--steer"></span>
          <span class="legend-label">转向损失 (Steer)</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot legend-dot--throttle"></span>
          <span class="legend-label">油门损失 (Throttle)</span>
        </div>
        <div class="legend-item">
          <span class="legend-dot legend-dot--accuracy"></span>
          <span class="legend-label">准确率</span>
        </div>
      </div>
    </div>
    <div ref="chartRef" class="chart-body"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import { useTrainingStore, type TrainHistoryEntry } from '@/stores/training'

const store = useTrainingStore()
const chartRef = ref<HTMLElement | null>(null)
let chartInstance: echarts.ECharts | null = null

/** 构建 ECharts option */
function buildOption(history: TrainHistoryEntry[]): echarts.EChartsOption {
  const showEmpty = history.length === 0

  const epochs = showEmpty
    ? [] as string[]
    : history.map(h => `E${h.epoch}`)

  const steerData = showEmpty
    ? []
    : history.map(h => h.steer_loss)

  const throttleData = showEmpty
    ? []
    : history.map(h => h.throttle_loss)

  const trainData = showEmpty ? [] : history.map(h => h.train_loss ?? null)
  const validationData = showEmpty ? [] : history.map(h => h.val_loss ?? null)
  const accuracyData = showEmpty ? [] : history.map(h => h.accuracy ?? null)

  return {
    grid: { top: 30, right: 50, bottom: 30, left: 55, containLabel: true },
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const epoch = params[0]?.axisValue ?? ''
        let html = `<b>${epoch}</b><br/>`
        params.forEach((p: any) => {
          html += `${p.marker} ${p.seriesName}: <b>${Number(p.value).toFixed(6)}</b><br/>`
        })
        return html
      },
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: epochs,
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: { color: '#64748B', fontSize: 10 },
    },
    yAxis: [
      {
        type: 'value',
        name: '转向损失',
        nameTextStyle: { color: '#64748B', fontSize: 10 },
        splitLine: { lineStyle: { color: '#F1F5F9' } },
        axisLabel: { color: '#64748B', fontSize: 10 },
      },
      {
        type: 'value',
        name: '油门损失',
        nameTextStyle: { color: '#64748B', fontSize: 10 },
        splitLine: { show: false },
        axisLabel: { color: '#64748B', fontSize: 10 },
      },
    ],
    series: [
      {
        name: '训练总损失', type: 'line', smooth: true, data: trainData,
        lineStyle: { color: '#334155', width: 2 }, itemStyle: { color: '#334155' },
      },
      {
        name: '验证总损失', type: 'line', smooth: true, data: validationData,
        lineStyle: { color: '#10b981', width: 2, type: 'dashed' }, itemStyle: { color: '#10b981' },
      },
      {
        name: '转向损失',
        type: 'line',
        smooth: true,
        data: steerData,
        lineStyle: { color: '#ffb95f', width: 3 },
        itemStyle: { color: '#ffb95f' },
        markPoint: steerData.length > 0 ? {
          data: [{
            type: 'min',
            name: '最佳',
            label: {
              show: true,
              formatter: '最佳',
              fontSize: 9,
              backgroundColor: '#855300',
              padding: [2, 4],
              borderRadius: 4,
              color: '#fff',
            },
            symbol: 'pin',
            symbolSize: 30,
          }],
        } : undefined,
      },
      {
        name: '油门损失',
        type: 'line',
        yAxisIndex: 1,
        smooth: true,
        data: throttleData,
        lineStyle: { color: '#2563eb', width: 3 },
        itemStyle: { color: '#2563eb' },
        markPoint: throttleData.length > 0 ? {
          data: [{
            type: 'min',
            name: '最佳',
            label: {
              show: true,
              formatter: '最佳',
              fontSize: 9,
              backgroundColor: '#2563eb',
              padding: [2, 4],
              borderRadius: 4,
              color: '#fff',
            },
            symbol: 'pin',
            symbolSize: 30,
          }],
        } : undefined,
      },
      {
        name: '准确率', type: 'line', yAxisIndex: 1, smooth: true,
        connectNulls: false, data: accuracyData,
        lineStyle: { color: '#8b5cf6', width: 2 }, itemStyle: { color: '#8b5cf6' },
      },
    ],
  }
}

function initChart() {
  if (!chartRef.value) return
  chartInstance = echarts.init(chartRef.value)
  chartInstance.setOption(buildOption(store.history))
}

/** 数据变化时刷新图表（notMerge=true 完全替换数据） */
function refreshChart() {
  if (!chartInstance) return
  chartInstance.setOption(buildOption(store.history), true)
}

// 监听 history 变化（length 改变 → 新的 epoch 数据到达）
watch(() => store.history.length, () => {
  refreshChart()
})

// 监听训练启动/停止（用于切换演示数据 ↔ 真实数据）
watch(() => store.running, () => {
  refreshChart()
})

function onResize() { chartInstance?.resize() }

onMounted(() => {
  initChart()
  window.addEventListener('resize', onResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', onResize)
  chartInstance?.dispose()
})
</script>

<style scoped>
.chart-card {
  background: var(--color-surface-container-lowest);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-xl);
  padding: var(--spacing-card);
  box-shadow: var(--shadow-card);
  height: 320px;
  display: flex;
  flex-direction: column;
}
.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.75rem;
  flex-shrink: 0;
}
.chart-title {
  font-size: var(--font-size-title);
  font-weight: 600;
  color: var(--color-text-main);
}
.chart-legend { display: flex; gap: 1rem; }
.legend-item { display: flex; align-items: center; gap: 0.5rem; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.legend-dot--steer { background: #ffb95f; }
.legend-dot--throttle { background: #2563eb; }
.legend-dot--train { background: #10b981; }
.legend-dot--accuracy { background: #8b5cf6; }
.legend-label {
  font-size: var(--font-size-label); font-weight: 700;
  color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em;
}
.chart-body { flex: 1; width: 100%; min-height: 0; }
</style>
