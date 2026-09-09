/**
 * main.ts — Vue 应用入口
 * 初始化 Vue、Router、Pinia，并启动 Mock 服务
 */
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
// 联调真实后端时注释掉下一行 import，取消 vite.config.ts 中 proxy 注释
import './styles/global.css'

// 全局错误捕获 — 将运行时错误显示在页面上便于调试
window.addEventListener('error', (e) => {
  console.error('[Global Error]', e.error)
  const appEl = document.getElementById('app')
  if (appEl) {
    appEl.innerHTML = `<div style="padding:2rem;color:red;font-family:monospace;">
      <h2>应用启动错误</h2>
      <pre>${e.error?.message || e.message}</pre>
      <pre>${e.error?.stack || ''}</pre>
    </div>`
  }
})

// 联调真实后端时注释掉下面这行
// try {
//   startMockServer()
// } catch (e: any) {
//   console.error('[Mock Server Error]', e)
// }

const app = createApp(App)

// Vue 全局错误处理
app.config.errorHandler = (err: any, instance, info) => {
  console.error('[Vue Error]', err, info)
  const appEl = document.getElementById('app')
  if (appEl && !appEl.textContent?.trim()) {
    appEl.innerHTML = `<div style="padding:2rem;color:red;font-family:monospace;">
      <h2>Vue 渲染错误</h2>
      <pre>${err?.message || err}</pre>
      <pre>${err?.stack || ''}</pre>
      <p>组件: ${instance?.$options?.name || instance?.$.type?.name || 'unknown'}</p>
    </div>`
  }
}

app.use(createPinia())
app.use(router)
app.mount('#app')
