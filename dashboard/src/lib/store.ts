import { create } from 'zustand'
import type { ComponentId, Config, EventRec, SessionStatus, Tick } from './types'
import { get } from './api'

const MAX_TICKS = 360
const MAX_EVENTS = 150

interface LiveStore {
  connected: boolean
  ready: boolean
  ticks: Tick[]
  events: EventRec[]
  status: SessionStatus | null
  config: Config | null
  selected: ComponentId
  showTruth: boolean
  theme: 'dark' | 'light'
  setSelected: (c: ComponentId) => void
  toggleTruth: () => void
  toggleTheme: () => void
  setStatus: (s: SessionStatus) => void
  connect: () => void
  refresh: () => void
}

let socket: WebSocket | null = null
let retry = 0

const savedTheme = (() => {
  try {
    return (localStorage.getItem('opsmind-theme') as 'dark' | 'light') || 'dark'
  } catch {
    return 'dark'
  }
})()
document.documentElement.dataset.theme = savedTheme

export const useLive = create<LiveStore>((set, getState) => ({
  connected: false,
  ready: false,
  ticks: [],
  events: [],
  status: null,
  config: null,
  selected: 'order',
  showTruth: true,
  theme: savedTheme,
  setSelected: (selected) => set({ selected }),
  toggleTruth: () => set((s) => ({ showTruth: !s.showTruth })),
  setStatus: (status) => set({ status }),
  toggleTheme: () =>
    set((s) => {
      const theme = s.theme === 'dark' ? 'light' : 'dark'
      document.documentElement.dataset.theme = theme
      try {
        localStorage.setItem('opsmind-theme', theme)
      } catch {
        /* ignore */
      }
      return { theme }
    }),
  refresh: () => {
    if (socket && socket.readyState === 1) socket.send('snapshot')
  },
  connect: () => {
    if (socket && socket.readyState <= 1) return
    if (!getState().config) get<Config>('/api/config').then((config) => set({ config })).catch(() => {})
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/live`)
    socket = ws
    ws.onopen = () => {
      retry = 0
      set({ connected: true })
    }
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data)
      if (msg.type === 'hello') {
        const ticks: Tick[] = msg.snapshot
        set({
          ticks,
          status: msg.status,
          ready: true,
          events: ticks.flatMap((t) => t.events).slice(-MAX_EVENTS),
        })
      } else if (msg.type === 'tick') {
        const t = msg as Tick
        set((s) => ({
          ticks: s.ticks.length >= MAX_TICKS ? [...s.ticks.slice(1), t] : [...s.ticks, t],
          events: t.events.length ? [...s.events, ...t.events].slice(-MAX_EVENTS) : s.events,
        }))
      } else if (msg.type === 'status') {
        set({ status: msg })
      }
    }
    ws.onclose = () => {
      set({ connected: false })
      socket = null
      retry = Math.min(retry + 1, 6)
      setTimeout(() => getState().connect(), 800 * retry)
    }
    ws.onerror = () => ws.close()
  },
}))

