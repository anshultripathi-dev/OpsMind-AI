import { CheckCircle2, ShieldAlert, BellRing, Radar, Siren, TrendingUp } from 'lucide-react'
import clsx from 'clsx'
import { Badge } from '@/components/ui'
import { useLive } from '@/lib/store'
import { COMPONENT_LABELS, FAULT_LABELS } from '@/lib/colors'
import { fmtClock, fmtDur, fmtPct } from '@/lib/format'

function Step({ label, time, done, tone }: { label: string; time?: string; done: boolean; tone: 'ok' | 'warn' | 'crit' | 'neutral' }) {
  const color = { ok: 'var(--ok)', warn: 'var(--warn)', crit: 'var(--crit)', neutral: 'var(--muted)' }[tone]
  return (
    <div className="flex min-w-[96px] flex-col gap-1">
      <div className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full" style={{ background: done ? color : 'transparent', border: `1.5px solid ${done ? color : 'var(--border-strong)'}`, boxShadow: done ? `0 0 8px ${color}` : undefined }} />
        <span className={clsx('text-[11.5px] font-medium', done ? 'text-fg' : 'text-muted')}>{label}</span>
      </div>
      <span className="pl-3.5 font-mono text-[11px] tnum text-muted">{done ? time ?? 'yes' : '—'}</span>
    </div>
  )
}

export function StatusHero() {
  const tick = useLive((s) => s.ticks[s.ticks.length - 1])
  if (!tick) return <div className="card skeleton h-[104px]" />
  const inc = tick.incident.current
  const state: 'incident' | 'warning' | 'healthy' = inc ? 'incident' : tick.risk.alert ? 'warning' : 'healthy'
  const cfg = {
    healthy: { tone: 'ok', Icon: CheckCircle2, title: 'All systems healthy', glow: 'var(--ok)' },
    warning: { tone: 'warn', Icon: BellRing, title: 'Early warning · incident likely within 2 minutes', glow: 'var(--warn)' },
    incident: { tone: 'crit', Icon: ShieldAlert, title: inc ? `${inc.id} · ${inc.type_name ?? tick.classifier.likely_fault_name}` : '', glow: 'var(--crit)' },
  }[state]
  const top = (inc?.rca?.[0] ?? tick.rca?.[0]) as { component: keyof typeof COMPONENT_LABELS; score: number } | undefined
  const conf = tick.classifier.likely_fault
  const lifecycle = inc ?? tick.incident.history[0]
  const startedT = inc?.started_at

  let subtitle: React.ReactNode
  if (state === 'healthy') {
    subtitle = (
      <>
        Risk {fmtPct(tick.risk.prob, 1)} · anomaly score {tick.anomaly.ratio.toFixed(2)}× of threshold · {tick.baseline.firing.length ? <span className="text-warn">{tick.baseline.firing.length} static alert rule firing</span> : 'no static alert rules firing'}
      </>
    )
  } else if (state === 'warning') {
    subtitle = <>The early-warning LSTM sees the {FAULT_LABELS[conf]?.toLowerCase()} precursor pattern — probability {fmtPct(tick.risk.prob)}. No user impact yet.</>
  } else {
    subtitle = (
      <>
        {top ? <>Most likely cause <b className="text-fg">{COMPONENT_LABELS[top.component]}</b> · </> : null}
        classifier confidence {fmtPct(tick.classifier.confidence)} · open for {fmtDur(tick.t - (startedT ?? tick.t))}
        {inc?.impact_at ? <span className="text-crit"> · users impacted</span> : <span className="text-ok"> · no user impact yet</span>}
      </>
    )
  }

  return (
    <div className="card relative overflow-hidden animate-fade-up">
      <div className="pointer-events-none absolute inset-0 opacity-60" style={{ background: `radial-gradient(900px 140px at 0% 0%, color-mix(in oklab, ${cfg.glow} 16%, transparent), transparent 70%)` }} />
      <div className="grid-bg pointer-events-none absolute inset-0 opacity-30 [mask-image:linear-gradient(90deg,black,transparent_70%)]" />
      <div className="relative flex flex-wrap items-center justify-between gap-x-8 gap-y-4 px-5 py-4">
        <div className="flex min-w-0 items-center gap-4">
          <div className="relative grid h-12 w-12 shrink-0 place-items-center rounded-2xl" style={{ background: `color-mix(in oklab, ${cfg.glow} 14%, transparent)`, color: cfg.glow }}>
            {state !== 'healthy' && <span className="absolute inset-0 rounded-2xl" style={{ border: `1.5px solid ${cfg.glow}`, animation: 'pulse-ring 1.8s ease-out infinite' }} />}
            <cfg.Icon size={24} />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-[19px] font-semibold tracking-tight">{cfg.title}</h2>
              {state === 'incident' && <Badge tone="crit" dot>Open</Badge>}
              {tick.pattern.novel && <Badge tone="warn">Unseen pattern</Badge>}
            </div>
            <p className="mt-0.5 text-[12.5px] text-fg-2">{subtitle}</p>
          </div>
        </div>

        <div className="flex items-start gap-5">
          <Step label="Early warning" done={!!lifecycle?.warning_at || state === 'warning'} time={lifecycle?.warning_at ? fmtClock(lifecycle.warning_at) : fmtClock(tick.t)} tone="warn" />
          <Step label="Detected" done={!!lifecycle} time={lifecycle ? fmtClock(lifecycle.detected_at) : undefined} tone="crit" />
          <Step label="User impact" done={!!lifecycle?.impact_at} time={lifecycle?.impact_at ? fmtClock(lifecycle.impact_at) : undefined} tone="crit" />
          <Step label="Static alert" done={!!lifecycle?.baseline_at} time={lifecycle?.baseline_at ? fmtClock(lifecycle.baseline_at) : undefined} tone="neutral" />
        </div>
      </div>
    </div>
  )
}

export { Radar, Siren, TrendingUp }
