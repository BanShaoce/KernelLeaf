/**
 * SystemChart.vue — CPU / GPU / 显存使用率实时曲线（ECharts）
 *
 * 复刻 reference/autodrive_dashboard_1/code.html 设计：
 * - CPU 使用率：绿色面积图 (#10B981)
 * - GPU 利用率：紫色折线 (#8B5CF6)
 * - 显存占用：橙色虚线 (#F97316)
 * - X 轴时间 HH:MM:SS，保留最近 10 分钟窗口（300 条 @ 2s）
 */
<template>
  <div class="chart-card">
    <div class="chart-header">
      <h3 class="chart-title">遥测与资源</h3>
      <div class="chart-legend">
        <div class="legend-item"><span class="legend-dot legend-dot--cpu"></span><span class="legend-label">CPU</span></div>
        <div class="legend-item"><span class="legend-dot legend-dot--gpu"></span><span class="legend-label">GPU</span></div>
        <div class="legend-item"><span class="legend-dot legend-dot--vram"></span><span class="legend-label">显存</span></div>
      </div>
    </div>
    <div ref="chartRef" class="chart-body"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import { useTrainingStore } from '@/stores/training'

const store = useTrainingStore()
const chartRef = ref<HTMLElement | null>(null)
let chartInstance: echarts.ECharts | null = null

/** 格式化时间戳 → HH:MM:SS */
function fmtTime(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString('zh-CN', { hour12: false })
}

function buildOption(): echarts.EChartsOption {
  const samples = store.systemHistory.slice(-300)
  const hasData = samples.length > 0

  const times = hasData
    ? samples.map((s: any) => fmtTime(s.timestamp || (Date.now() / 1000)))
    : []

  const cpuData = hasData ? samples.map((s: any) => s.cpu_percent ?? 0) : []
  const gpuData = hasData ? samples.map((s: any) => s.gpus?.[0]?.util_percent ?? 0) : []
  const vramData = hasData ? samples.map((s: any) => s.gpus?.[0]?.mem_percent ?? 0) : []

  // X 轴标签间隔：约每 30s 一个标签
  const labelInterval = hasData ? Math.max(Math.floor(times.length / 10), 1) : 1

  return {
    grid: { top: 30, right: 30, bottom: 30, left: 50, containLabel: true },
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const time = params[0]?.axisValue ?? ''
        let html = `<b>${time}</b><br/>`
        params.forEach((p: any) => {
          html += `${p.marker} ${p.seriesName}: <b>${Number(p.value).toFixed(1)}%</b><br/>`
        })
        return html
      },
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: times,
      axisLine: { lineStyle: { color: '#E2E8F0' } },
      axisLabel: {
        color: '#64748B',
        fontSize: 10,
        interval: labelInterval,
        rotate: hasData ? 25 : 0,
      },
    },
    yAxis: {
      type: 'value',
      max: 100,
      splitLine: { lineStyle: { color: '#F1F5F9' } },
      axisLabel: { color: '#64748B', fontSize: 10, formatter: '{value}%' },
    },
    series: [
      {
        name: 'CPU',
        type: 'line',
        areaStyle: { color: 'rgba(16, 185, 129, 0.1)' },
        data: cpuData,
        lineStyle: { color: '#10B981', width: 1 },
        itemStyle: { color: '#10B981' },
        showSymbol: false,
        smooth: true,
      },
      {
        name: 'GPU',
        type: 'line',
        data: gpuData,
        lineStyle: { color: '#8B5CF6', width: 2 },
        itemStyle: { color: '#8B5CF6' },
        showSymbol: false,
        smooth: true,
      },
      {
        name: '显存占用',
        type: 'line',
        data: vramData,
        lineStyle: { color: '#F97316', width: 2, type: 'dashed' },
        itemStyle: { color: '#F97316' },
        showSymbol: false,
        smooth: true,
      },
    ],
  }
}

function initChart() {
  if (!chartRef.value) return
  chartInstance = echarts.init(chartRef.value)
  chartInstance.setOption(buildOption())
}

function refreshChart() {
  if (!chartInstance) return
  chartInstance.setOption(buildOption(), true)
}

// 监听 systemHistory 引用变化（数据已改为整体替换）
watch(() => store.systemHistory, () => {
  refreshChart()
}, { deep: false })

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
.chart-title { font-size: var(--font-size-title); font-weight: 600; color: var(--color-text-main); }
.chart-legend { display: flex; gap: 1rem; }
.legend-item { display: flex; align-items: center; gap: 0.5rem; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.legend-dot--cpu { background: rgba(16,185,129,0.3); border: 2px solid #10B981; }
.legend-dot--gpu { background: #8B5CF6; }
.legend-dot--vram { border: 2px dashed #F97316; background: transparent; }
.legend-label {
  font-size: var(--font-size-label); font-weight: 700;
  color: var(--color-text-muted); text-transform: uppercase; letter-spacing: 0.05em;
}
.chart-body { flex: 1; width: 100%; min-height: 0; }
</style>
