import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { describe, expect, it } from 'vitest'

import TrainingHistory from './TrainingHistory.vue'
import { useTrainingStore } from '@/stores/training'

describe('TrainingHistory', () => {
  it('renders persisted KernelLeaf epoch metrics', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useTrainingStore()
    store.history = [{
      epoch: 1,
      lr: 0.001,
      steer_loss: 0.25,
      throttle_loss: 0.05,
      best_steer_loss: 0.25,
      best_throttle_loss: 0.05,
      epoch_time: 1.2,
      n_batches: 10,
      is_best_steer: true,
      is_best_throttle: true,
    }]

    const wrapper = mount(TrainingHistory, { global: { plugins: [pinia] } })
    expect(wrapper.text()).toContain('001')
    expect(wrapper.text()).toContain('0.250000')
    expect(wrapper.text()).toContain('最佳')
  })
})
