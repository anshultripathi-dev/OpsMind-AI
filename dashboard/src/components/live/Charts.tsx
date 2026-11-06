import { useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Activity, LineChart as LineIcon } from 'lucide-react'
import clsx from 'clsx'
import { Card, CardHeader, ChartTooltip, Segmented } from '@/components/ui'
import { useLive } from '@/lib/store'
import { COMPONENT_COLORS, COMPONENT_LABELS } from '@/lib/colors'
import { fmtClock, fmtClockShort, fmtPct } from '@/lib/format'
import type { ComponentId } from '@/lib/types'
import { EventLegend, overlays } from './chartUtils'

const RANGES = [
  { value: 60, label: '10 min' },
  { value: 180, label: '30 min' },
  { value: 360, label: '1 h' },
]
const METRIC_TABS = [
  { value: 'cpu', label: 'CPU' },
  { value: 'mem', label: 'Memory' },
  { value: 'lat_p95', label: 'p95 latency' },
  { value: 'err_rate', label: 'Errors' },
  { value: 'rps', label: 'Throughput' },
]
const UNITS: Record<string, string> = { cpu: '%', mem: '%', lat_p95: ' ms', err_rate: '%', rps: ' req/s' }
const COMPONENTS: ComponentId[] = ['gateway', 'auth', 'order', 'payment', 'postgres', 'redis']

const axisProps = { tick: { fill: 'var(--muted)', fontSize: 10.5 }, axisLine: false, tickLine: false } as const

export function RiskTimeline() {
  const { ticks, events, showTruth } = useLive()
  const [range, setRange] = useState(180)
  const data = useMemo(() => ticks.slice(-range).map((k) => ({ t: k.t, risk: k.risk.prob, anomaly: Math.max(k.anomaly.ratio, -0.2) })), [ticks, range])
  const view = ticks.slice(-range)
  const thr = ticks[ticks.length - 1]?.risk.threshold ?? 0.3
  const ov = useMemo(() => overlays(view, events, showTruth), [view, events, showTruth])
  const xProps = { dataKey: 't', type: 'number' as const, domain: ['dataMin', 'dataMax'] as [string, string], tickFormatter: fmtClockShort, minTickGap: 60, ...axisProps }

  return (
    <Card className="animate-fade-up">
      <CardHeader icon={<Activity size={15} />} title="Early warning & anomaly timeline" subtitle="Risk of an incident in the next 2 minutes, and distance from normal behaviour" actions={<Segmented size="sm" value={range} onChange={setRange} options={RANGES} />} />
      <div className="px-2">
        <div className="px-2 pt-1 text-[10.5px] font-semibold uppercase tracking-wider text-muted">Incident risk</div>
        <ResponsiveContainer width="100%" height={118}>
          <AreaChart data={data} margin={{ top: 6, right: 10, left: -18, bottom: 0 }}>
            <defs>
              <linearGradient id="riskfill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0" stopColor="var(--warn)" stopOpacity={0.45} />
                <stop offset="1" stopColor="var(--warn)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} />
            <XAxis {...xProps} hide />
            <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={(v) => fmtPct(v)} {...axisProps} />
            {ov}
            <ReferenceLine y={thr} stroke="var(--muted)" strokeDasharray="4 4" label={{ value: 'alert', position: 'insideTopRight', fill: 'var(--muted)', fontSize: 10 }} />
            <Tooltip content={<ChartTooltip labelFormatter={(l: number) => fmtClock(l)} valueFormatter={(v: number) => fmtPct(v, 1)} />} />
            <Area name="Risk" type="monotone" dataKey="risk" stroke="var(--warn)" strokeWidth={2} fill="url(#riskfill)" isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
        <div className="px-2 pt-1 text-[10.5px] font-semibold uppercase tracking-wider text-muted">Anomaly score ÷ threshold</div>
        <ResponsiveContainer width="100%" height={118}>
          <LineChart data={data} margin={{ top: 6, right: 10, left: -18, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis {...xProps} />
            <YAxis domain={[-0.2, 'auto']} tickFormatter={(v) => `${v.toFixed(1)}×`} {...axisProps} />
            {ov}
            <ReferenceLine y={1} stroke="var(--crit)" strokeDasharray="4 4" strokeOpacity={0.7} />
            <Tooltip content={<ChartTooltip labelFormatter={(l: number) => fmtClock(l)} valueFormatter={(v: number) => `${v.toFixed(2)}×`} />} />
            <Line name="Anomaly" type="monotone" dataKey="anomaly" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <div className="px-3 pb-3 pt-1">
          <EventLegend />
        </div>
      </div>
    </Card>
  )
}

export function MetricExplorer() {
  const { ticks, events, showTruth, selected, setSelected } = useLive()
  const [metric, setMetric] = useState('lat_p95')
  const [range, setRange] = useState(180)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const view = useMemo(() => ticks.slice(-range), [ticks, range])
  const log = metric === 'lat_p95'
  const data = useMemo(
    () =>
      view.map((k) => {
        const row: Record<string, number> = { t: k.t }
        for (const c of COMPONENTS) {
          const v = k.sample[c]?.[metric]
          if (v !== undefined && !(metric === 'err_rate' && c === 'redis')) row[c] = log ? Math.max(v, 0.5) : v
        }
        return row
      }),
    [view, metric, log],
  )
  const ov = useMemo(() => overlays(view, events, showTruth), [view, events, showTruth])
  const toggle = (c: string) => setHidden((h) => { const n = new Set(h); n.has(c) ? n.delete(c) : n.add(c); return n })

  return (
    <Card className="animate-fade-up">
      <CardHeader
        icon={<LineIcon size={15} />}
        title="Metric explorer"
        subtitle="Raw Prometheus-style series for every component"
        actions={<Segmented size="sm" value={range} onChange={setRange} options={RANGES} />}
      />
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 pb-1">
        <Segmented size="sm" value={metric} onChange={setMetric} options={METRIC_TABS} />
        <div className="flex flex-wrap gap-1">
          {COMPONENTS.map((c) => (
            <button key={c} onClick={() => toggle(c)} onDoubleClick={() => setSelected(c)} className={clsx('flex items-center gap-1.5 rounded-md border px-1.5 py-[2px] text-[11px] transition-all', hidden.has(c) ? 'border-transparent text-muted opacity-50' : 'border-line bg-surface-2 text-fg-2', selected === c && !hidden.has(c) && 'border-accent/50')} title="Click to toggle · double-click to focus">
              <span className="h-2 w-2 rounded-full" style={{ background: COMPONENT_COLORS[c] }} />
              {COMPONENT_LABELS[c]}
            </button>
          ))}
        </div>
      </div>
      <div className="px-2">
        <ResponsiveContainer width="100%" height={270}>
          <LineChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tickFormatter={fmtClockShort} minTickGap={60} {...axisProps} />
            <YAxis scale={log ? 'log' : 'auto'} domain={log ? [1, 'auto'] : metric === 'cpu' || metric === 'mem' ? [0, 100] : [0, 'auto']} allowDataOverflow={log} tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${Math.round(v)}`)} {...axisProps} />
            {ov}
            {metric === 'lat_p95' && <ReferenceLine y={500} stroke="var(--muted)" strokeDasharray="4 4" label={{ value: 'static alert 500 ms', position: 'insideTopLeft', fill: 'var(--muted)', fontSize: 10 }} />}
            {(metric === 'cpu' || metric === 'mem') && <ReferenceLine y={90} stroke="var(--muted)" strokeDasharray="4 4" label={{ value: 'static alert 90 %', position: 'insideTopLeft', fill: 'var(--muted)', fontSize: 10 }} />}
            <Tooltip content={<ChartTooltip labelFormatter={(l: number) => fmtClock(l)} valueFormatter={(v: number) => `${v.toFixed(v < 10 ? 2 : 0)}${UNITS[metric]}`} />} />
            {COMPONENTS.filter((c) => !hidden.has(c) && !(metric === 'err_rate' && c === 'redis')).map((c) => (
              <Line key={c} name={COMPONENT_LABELS[c]} dataKey={c} type="monotone" stroke={COMPONENT_COLORS[c]} strokeWidth={selected === c ? 2.6 : 1.5} strokeOpacity={selected === c ? 1 : 0.75} dot={false} isAnimationActive={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
        <div className="px-3 pb-3 pt-1">
          <EventLegend />
        </div>
      </div>
    </Card>
  )
}
