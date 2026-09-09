/**
 * AppHeader.vue — 顶部导航栏
 *
 * 复刻 reference 中的 TopAppBar 设计：
 * - 左侧 Logo + Tab 导航
 * - 右侧操作图标 + 用户头像
 * - 当前路由高亮（底部蓝色指示条）
 */
<template>
  <header class="app-header">
    <!-- 左侧 -->
    <div class="header-left">
      <!-- Logo -->
      <span class="header-logo">
        <span class="material-symbols-outlined logo-icon">memory</span>
        KernelLeaf AutoDrive
      </span>
      <!-- Tab 导航 -->
      <nav class="header-nav">
        <router-link
          v-for="tab in tabs"
          :key="tab.path"
          :to="tab.path"
          class="nav-tab"
          active-class="nav-tab--active"
        >
          {{ tab.label }}
        </router-link>
      </nav>
    </div>

    <!-- 右侧 -->
    <div class="header-right">
      <!-- 系统状态指示 -->
      <div class="status-badge">
        <span
          class="status-dot"
          :class="{
            'status-dot--running': store.status === 'running',
            'status-dot--paused': store.status === 'paused',
          }"
        ></span>
        <span class="status-text">
          {{ statusLabel }}
        </span>
      </div>

      <!-- 通知 -->
      <button class="icon-btn" title="通知">
        <span class="material-symbols-outlined">notifications</span>
      </button>
      <!-- 设置 -->
      <button class="icon-btn" title="设置">
        <span class="material-symbols-outlined">settings</span>
      </button>
      <!-- 用户头像 -->
      <div class="avatar">
        <span class="material-symbols-outlined">person</span>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
/**
 * 顶部导航栏组件
 * 展示 4 个 Tab 页导航 + 系统状态指示
 */
import { computed } from 'vue'
import { useTrainingStore } from '@/stores/training'

const store = useTrainingStore()

/** Tab 列表 */
const tabs = [
  { path: '/training', label: '训练监控' },
  { path: '/gradcam', label: 'Grad-CAM' },
  { path: '/validation', label: '验证评估' },
  { path: '/data', label: '数据管理' },
]

/** 状态文字 */
const statusLabel = computed(() => {
  switch (store.status) {
    case 'running': return '运行中'
    case 'paused': return '已暂停'
    default: return '空闲'
  }
})
</script>

<style scoped>
/* ---------- 容器 ---------- */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  height: 4rem;
  padding: 0 var(--spacing-gutter);
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
  z-index: 50;
}

/* ---------- 左侧 ---------- */
.header-left {
  display: flex;
  align-items: center;
  gap: 2rem;
}

.header-logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: var(--font-size-title);
  font-weight: 700;
  color: var(--color-primary);
  white-space: nowrap;
}
.logo-icon {
  font-size: 1.5rem;
}

/* ---------- Tab 导航 ---------- */
.header-nav {
  display: flex;
  align-items: center;
  gap: 1.5rem;
}

.nav-tab {
  position: relative;
  text-decoration: none;
  font-size: var(--font-size-label);
  font-weight: 500;
  color: var(--color-text-muted);
  padding-bottom: 0.25rem;
  transition: color 0.2s;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.nav-tab:hover {
  color: var(--color-primary);
}

/* 当前激活 Tab — 底部蓝色条 + 加粗 */
.nav-tab--active {
  color: var(--color-primary);
  font-weight: 700;
}
.nav-tab--active::after {
  content: '';
  position: absolute;
  bottom: -1.2rem;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--color-primary);
  border-radius: 1px;
}

/* ---------- 右侧 ---------- */
.header-right {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.75rem;
  border-radius: var(--radius-full);
  background: var(--color-surface-container-low);
  font-size: var(--font-size-body-sm);
  font-weight: 500;
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
  font-size: var(--font-size-label);
  font-weight: 700;
  color: var(--color-text-main);
}

/* ---------- 图标按钮 ---------- */
.icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border: none;
  border-radius: var(--radius-full);
  background: transparent;
  color: var(--color-on-surface-variant);
  cursor: pointer;
  transition: background-color 0.2s;
}
.icon-btn:hover {
  background: var(--color-surface-container);
}

/* ---------- 头像 ---------- */
.avatar {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  border-radius: var(--radius-full);
  background: var(--color-primary-fixed);
  color: var(--color-primary);
  border: 1px solid var(--color-outline-variant);
  overflow: hidden;
  font-size: 1.25rem;
}

/* ---------- 响应式：小屏隐藏导航文字 ---------- */
@media (max-width: 768px) {
  .header-nav {
    display: none;
  }
}
</style>
