/**
 * App.vue — 根组件
 * 提供全局 Toast 容器 + Router View
 */
<template>
  <div id="app-root">
    <!-- Toast 通知容器 -->
    <div class="toast-container">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        :class="['toast-item', toast.type]"
      >
        {{ toast.message }}
      </div>
    </div>
    <!-- 路由视图 -->
    <router-view />
  </div>
</template>

<script setup lang="ts">
/**
 * 全局 Toast 管理
 * 通过 provide/inject 或直接 emit 供子组件调用
 */
import { ref, provide } from 'vue'

interface Toast {
  id: number
  message: string
  type: 'error' | 'success' | 'warning' | 'info'
}

const toasts = ref<Toast[]>([])
let toastId = 0

/**
 * 显示一条 toast 通知
 * @param message - 通知内容
 * @param type - 类型（error / success / warning / info）
 * @param duration - 自动消失时间（毫秒），默认 4000
 */
function showToast(message: string, type: Toast['type'] = 'info', duration = 4000) {
  const id = ++toastId
  toasts.value.push({ id, message, type })
  setTimeout(() => {
    toasts.value = toasts.value.filter(t => t.id !== id)
  }, duration)
}

// 将 showToast 注入全局，子组件可通过 inject 使用
provide('showToast', showToast)
</script>
