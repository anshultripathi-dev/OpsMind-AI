const BASE = 9 * 3600 // simulated wall clock starts at 09:00:00

export function fmtClock(t: number): string {
  const s = Math.max(0, Math.floor(t + BASE))
  const h = Math.floor(s / 3600) % 24
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return [h, m, sec].map((v) => String(v).padStart(2, '0')).join(':')
}

export function fmtClockShort(t: number): string {
  return fmtClock(t).slice(0, 5)
}

export function fmtDur(sec: number): string {
  const s = Math.abs(Math.round(sec))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  return r ? `${m}m ${String(r).padStart(2, '0')}s` : `${m}m`
}

export function fmtPct(v: number, digits = 0): string {
  return `${(v * 100).toFixed(digits)}%`
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return v.toFixed(digits)
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}
