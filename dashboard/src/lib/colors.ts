import type { ComponentId, FaultType } from './types'

export const COMPONENT_COLORS: Record<ComponentId, string> = {
  gateway: '#818cf8',
  auth: '#2dd4bf',
  order: '#fbbf24',
  payment: '#f472b6',
  postgres: '#60a5fa',
  redis: '#c084fc',
}

export const FAULT_COLORS: Record<string, string> = {
  cpu_saturation: '#fb923c',
  memory_leak: '#facc15',
  db_slowdown: '#60a5fa',
  network_latency: '#22d3ee',
  service_crash: '#fb7185',
  cache_failure: '#c084fc',
  normal: '#64748b',
}

export const FAULT_LABELS: Record<string, string> = {
  cpu_saturation: 'CPU saturation',
  memory_leak: 'Memory leak',
  db_slowdown: 'DB slowdown',
  network_latency: 'Network latency',
  service_crash: 'Service crash',
  cache_failure: 'Cache failure',
  normal: 'Normal',
}

export const COMPONENT_LABELS: Record<ComponentId, string> = {
  gateway: 'API Gateway',
  auth: 'Auth',
  order: 'Order',
  payment: 'Payment',
  postgres: 'PostgreSQL',
  redis: 'Redis',
}

export const CLUSTER_COLORS = [
  '#818cf8', '#2dd4bf', '#fbbf24', '#f472b6', '#60a5fa', '#c084fc',
  '#fb923c', '#34d399', '#f87171', '#a3e635', '#38bdf8', '#e879f9', '#94a3b8',
]

export function healthTone(dev: number): 'ok' | 'warn' | 'crit' {
  if (dev < 0.45) return 'ok'
  if (dev < 0.72) return 'warn'
  return 'crit'
}

export const TONE_VAR: Record<'ok' | 'warn' | 'crit' | 'info', string> = {
  ok: 'var(--ok)',
  warn: 'var(--warn)',
  crit: 'var(--crit)',
  info: 'var(--info)',
}

export const FAULT_ORDER: FaultType[] = [
  'cpu_saturation',
  'memory_leak',
  'db_slowdown',
  'network_latency',
  'service_crash',
  'cache_failure',
]
