import { useMemo } from 'react'
import { Timer, Waves, Gauge as GaugeIcon, Hourglass } from 'lucide-react'
import { Badge, Card, Gauge, Spark } from '@/components/ui'
import { useLive } from '@/lib/store'
import { fmtDur, fmtPct } from '@/lib/format'

function Shell({ title, icon, children, badge }: { title: string; icon: React.ReactNode; children: React.ReactNode; badge?: React.ReactNode }) {
  return (
    <Card className="flex flex-col p-4 animate-fade-up">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2 text-muted">
          {icon}
          <span className="eyebrow">{title}</span>
        </div>
        {badge}
      </div>
      {children}
    </Card>
  )
}

export function KpiRow() {
  const ticks = useLive((s) => s.ticks)
  const last = ticks[ticks.length - 1]
  const recent = useMemo(() => ticks.slice(-60), [ticks])
  if (!last) return <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">{[0, 1, 2, 3].map((i) => <div key={i} className="card skeleton h-[150px]" />)}</div>

  const { risk, anomaly, slo } = last
  const riskTone = risk.prob >= risk.threshold ? 'crit' : risk.prob >= risk.threshold * 0.45 ? 'warn' : 'ok'
  const anTone = anomaly.flagged ? 'crit' : anomaly.ratio > 0.6 ? 'warn' : 'ok'
  const lifecycle = last.incident.current ?? last.incident.history[0]

  return (
    <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
      <Shell title="Incident risk · next 2 min" icon={<Hourglass size={13} />} badge={<Badge tone={riskTone} dot>{risk.alert ? 'Alert' : 'Calm'}</Badge>}>
        <div className="flex items-end justify-between gap-3">
          <Gauge value={risk.prob} threshold={risk.threshold} tone={riskTone} size={118}>
            <span className="text-[24px] font-semibold tracking-tight tnum">{fmtPct(risk.prob)}</span>
          </Gauge>
          <div className="min-w-0 flex-1 pb-1">
            <Spark data={recent.map((t) => t.risk.prob)} min={0} max={1} color={`var(--${riskTone})`} height={44} threshold={risk.threshold} />
          </div>
        </div>
        <p className="mt-2 text-[11.5px] text-muted">LSTM over the last 5 min · alert ≥ {fmtPct(risk.threshold)}</p>
      </Shell>

      <Shell title="Anomaly score" icon={<Waves size={13} />} badge={<Badge tone={anTone} dot>{anomaly.flagged ? 'Anomalous' : 'Normal'}</Badge>}>
        <div className="flex items-baseline gap-1.5">
          <span className="text-[30px] font-semibold leading-none tracking-tight tnum">{anomaly.ratio.toFixed(2)}</span>
          <span className="text-[13px] text-muted">× threshold</span>
        </div>
        <div className="mt-2.5">
          <Spark data={recent.map((t) => t.anomaly.ratio)} min={-0.2} max={Math.max(2, ...recent.map((t) => t.anomaly.ratio))} color={`var(--${anTone})`} height={44} threshold={1} />
        </div>
        <p className="mt-2 text-[11.5px] text-muted">One-Class SVM on normal behaviour · fires at 1.0×</p>
      </Shell>

      <Shell title="Gateway SLO" icon={<GaugeIcon size={13} />} badge={<Badge tone={slo.impacted ? 'crit' : 'ok'} dot>{slo.impacted ? 'Breached' : 'Within SLO'}</Badge>}>
        <div className="flex items-end gap-5">
          <div>
            <div className="eyebrow">p95 latency</div>
            <div className="flex items-baseline gap-1"><span className="text-[26px] font-semibold leading-none tracking-tight tnum" style={{ color: slo.lat > 200 ? 'var(--crit)' : undefined }}>{Math.round(slo.lat)}</span><span className="text-[12px] text-muted">ms</span></div>
          </div>
          <div>
            <div className="eyebrow">errors</div>
            <div className="flex items-baseline gap-1"><span className="text-[26px] font-semibold leading-none tracking-tight tnum" style={{ color: slo.err > 3 ? 'var(--crit)' : undefined }}>{slo.err.toFixed(1)}</span><span className="text-[12px] text-muted">%</span></div>
          </div>
        </div>
        <div className="mt-2.5">
          <Spark data={recent.map((t) => Math.min(t.slo.lat, 1200))} min={0} color={slo.impacted ? 'var(--crit)' : 'var(--info)'} height={44} threshold={200} />
        </div>
        <p className="mt-2 text-[11.5px] text-muted">SLO: p95 &lt; 200 ms and errors &lt; 3 %</p>
      </Shell>

      <Shell title="Warning lead time" icon={<Timer size={13} />} badge={lifecycle ? <Badge tone="neutral">{lifecycle.id}</Badge> : undefined}>
        {lifecycle && lifecycle.impact_at ? (
          <div className="space-y-3">
            <LeadRow label="OpsMind early warning" value={lifecycle.lead_s} tone="accent" />
            <LeadRow label="Static threshold alert" value={lifecycle.baseline_lead_s} tone="neutral" />
            <p className="text-[11.5px] text-muted">Positive = alerted before users were affected.</p>
          </div>
        ) : (
          <div className="flex h-[112px] flex-col items-start justify-center gap-1.5">
            <span className="text-[13px] font-medium text-fg-2">{lifecycle ? 'No user impact in this incident' : 'No incidents yet'}</span>
            <span className="max-w-[30ch] text-[12px] text-muted">Inject a fault from the chaos console to see how much earlier OpsMind warns than static alerts.</span>
          </div>
        )}
      </Shell>
    </div>
  )
}

function LeadRow({ label, value, tone }: { label: string; value: number | null; tone: 'accent' | 'neutral' }) {
  const has = value !== null
  const good = has && value! > 0
  const color = tone === 'accent' ? 'var(--accent)' : 'var(--muted)'
  const w = has ? Math.min(Math.abs(value!) / 180, 1) * 100 : 0
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[12px]">
        <span className="text-fg-2">{label}</span>
        <span className="font-semibold tnum" style={{ color: !has ? 'var(--muted)' : good ? (tone === 'accent' ? 'var(--ok)' : 'var(--fg)') : 'var(--crit)' }}>
          {!has ? 'never fired' : good ? `+${fmtDur(value!)} early` : `${fmtDur(value!)} late`}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.max(w, has ? 4 : 0)}%`, background: !has ? 'transparent' : good ? color : 'var(--crit)' }} />
      </div>
    </div>
  )
}
