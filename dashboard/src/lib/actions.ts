import { post } from './api'
import { useLive } from './store'
import type { SessionStatus } from './types'

const apply = (s: SessionStatus) => useLive.getState().setStatus(s)

export const actions = {
  pause: () => post<SessionStatus>('/api/session/control', { action: 'pause' }).then(apply),
  resume: () => post<SessionStatus>('/api/session/control', { action: 'resume' }).then(apply),
  reset: () =>
    post<SessionStatus>('/api/session/control', { action: 'reset' }).then((s) => {
      apply(s)
      useLive.getState().refresh()
    }),
  speed: (speed: number) => post<SessionStatus>('/api/session/speed', { speed }).then(apply),
  replay: (run_id: string) =>
    post<SessionStatus>('/api/session/replay', { run_id }).then((s) => {
      apply(s)
      useLive.getState().refresh()
    }),
  inject: (type: string, target: string | null, duration: number, intensity: number) =>
    post('/api/faults/inject', { type, target, duration, intensity }),
  spike: (factor: number, hold: number) => post('/api/faults/spike', { factor, hold }),
  clear: () => post<SessionStatus>('/api/faults/clear').then(apply),
  scenario: (id: string) => post<SessionStatus>('/api/scenarios/run', { id }).then(apply),
}
