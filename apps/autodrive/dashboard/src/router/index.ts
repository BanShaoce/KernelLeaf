/**
 * router/index.ts — Vue Router 配置
 * 4 个 Tab 页面对应 4 个路由
 */
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/training',
  },
  {
    path: '/training',
    name: 'TrainingMonitor',
    component: () => import('@/views/TrainingMonitor.vue'),
    meta: { title: '训练监控' },
  },
  {
    path: '/gradcam',
    name: 'GradCam',
    component: () => import('@/views/GradCam.vue'),
    meta: { title: 'Grad-CAM' },
  },
  {
    path: '/validation',
    name: 'Validation',
    component: () => import('@/views/Validation.vue'),
    meta: { title: '验证评估' },
  },
  {
    path: '/data',
    name: 'DataManagement',
    component: () => import('@/views/DataManagement.vue'),
    meta: { title: '数据管理' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
