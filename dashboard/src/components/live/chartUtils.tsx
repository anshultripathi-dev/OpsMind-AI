import { ReferenceArea, ReferenceLine } from 'recharts'
import type { EventRec, Tick } from '@/lib/types'

export const EVENT_STYLE: Record<string, { color: string; label: string }> = {
  early_warning: { color: 'var(--warn)', label: 'Early warning' },
  detected: { color: 'var(--accent)', label: 'Detected' },
  threshold_alert: { color: 'var(--muted)', label: 'Static alert' },
  impact: { color: 'var(--crit)', label: 'User impact' },
}

export function truthBands(ticks: Tick[]): [number, number][] {
  const out: [number, number][] = []
  let start: number | null = null
  for (const k of ticks) {
    const on = (k.truth?.faults?.length ?? 0) > 0
    if (on && start === null) start = k.t
    if (!on && start !== null) {
      out.push([start, k.t])
      start = null
    }
  }
  if (start !== null && ticks.length) out.push([start, ticks[ticks.length - 1].t])
  return out
}

export function overlays(ticks: Tick[], events: EventRec[], showTruth: boolean) {
  if (!ticks.length) return []
  const t0 = ticks[0].t
  const items: React.ReactNode[] = []
  if (showTruth)
    truthBands(ticks).forEach(([a, b], i) => items.push(<ReferenceArea key={`b${i}`} x1={a} x2={b} fill="var(--crit)" fillOpacity={0.07} stroke="var(--crit)" strokeOpacity={0.25} strokeDasharray="3 4" ifOverflow="hidden" />))
  events
    .filter((e) => EVENT_STYLE[e.kind] && e.t >= t0)
    .forEach((e, i) => items.push(<ReferenceLine key={`e${i}`} x={e.t} stroke={EVENT_STYLE[e.kind].color} strokeWidth={1.4} strokeDasharray={e.kind === 'threshold_alert' ? '2 3' : undefined} ifOverflow="hidden" />))
  return items
}

export function EventLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 text-[11px] text-muted">
      {Object.values(EVENT_STYLE).map((s) => (
        <span key={s.label} className="flex items-center gap-1.5">
          <span className="h-3 w-[2px] rounded" style={{ background: s.color }} />
          {s.label}
        </span>
      ))}
      <span className="flex items-center gap-1.5">
        <span className="h-3 w-3 rounded-sm border border-crit/40 bg-crit-soft" />
        Injected fault
      </span>
    </div>
  )
}
