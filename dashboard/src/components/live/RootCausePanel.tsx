import { useState } from 'react'
import { Crosshair, CheckCircle2, XCircle, ChevronDown, Search } from 'lucide-react'
import clsx from 'clsx'
import { Badge, Card, CardHeader, Empty } from '@/components/ui'
import { useLive } from '@/lib/store'
import { COMPONENT_COLORS, COMPONENT_LABELS, FAULT_LABELS } from '@/lib/colors'
import { fmtPct } from '@/lib/format'

export function RootCausePanel() {
  const { ticks, showTruth, toggleTruth } = useLive()
  const last = ticks[ticks.length - 1]
  const [open, setOpen] = useState<string | null>(null)
  const active = !!last?.rca && (last.anomaly.flagged || last.risk.alert || last.risk.prob > 0.3)
  const rca = active ? last!.rca!.slice(0, 4) : []
  const truth = showTruth ? last?.truth?.faults?.[0] : undefined
  const verdict = truth && rca.length ? (rca[0].component === truth.target ? 'hit' : rca.slice(0, 3).some((r) => r.component === truth.target) ? 'top3' : 'miss') : null
  const expanded = open ?? rca[0]?.component

  return (
    <Card className="flex h-full flex-col animate-fade-up">
      <CardHeader
        icon={<Crosshair size={15} />}
        title="Root-cause ranking"
        subtitle="SHAP attribution + dependency-graph walk"
        actions={
          <button onClick={toggleTruth} className={clsx('rounded-md px-2 py-1 text-[11px] font-medium transition-colors', showTruth ? 'bg-accent-soft text-accent' : 'text-muted hover:bg-surface-3')}>
            Ground truth
          </button>
        }
      />
      {!active ? (
        <Empty icon={<Search size={18} />} title="Nothing to diagnose" hint="Root-cause ranking activates the moment an anomaly or an early warning is raised." className="flex-1" />
      ) : (
        <div className="flex flex-1 flex-col gap-2 px-4 pb-4">
          <div className="flex items-center justify-between rounded-lg bg-surface-2 px-3 py-2">
            <div>
              <div className="eyebrow">Likely fault</div>
              <div className="text-[13.5px] font-semibold">{FAULT_LABELS[last!.classifier.likely_fault] ?? last!.classifier.likely_fault_name}</div>
            </div>
            <div className="text-right">
              <div className="eyebrow">Pattern</div>
              <div className="text-[12.5px] font-medium text-fg-2">{last!.pattern.name}</div>
            </div>
          </div>

          {rca.map((r, i) => {
            const isOpen = expanded === r.component
            return (
              <div key={r.component} className={clsx('rounded-lg border transition-colors', i === 0 ? 'border-crit/40 bg-crit-soft/40' : 'border-line bg-surface-2/60')}>
                <button className="flex w-full items-center gap-3 px-3 py-2.5 text-left" onClick={() => setOpen(isOpen ? '' : r.component)}>
                  <span className={clsx('grid h-6 w-6 shrink-0 place-items-center rounded-full text-[11px] font-bold', i === 0 ? 'bg-crit text-white' : 'bg-surface-3 text-fg-2')}>{i + 1}</span>
                  <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: COMPONENT_COLORS[r.component] }} />
                  <span className="flex-1 text-[13px] font-medium">{COMPONENT_LABELS[r.component]}</span>
                  <span className="w-12 text-right text-[13px] font-semibold tnum">{fmtPct(r.score)}</span>
                  <ChevronDown size={14} className={clsx('text-muted transition-transform', isOpen && 'rotate-180')} />
                </button>
                <div className="px-3 pb-2.5">
                  <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
                    <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.max(r.score * 100, 2)}%`, background: i === 0 ? 'var(--crit)' : COMPONENT_COLORS[r.component] }} />
                  </div>
                  {isOpen && (
                    <ul className="mt-2.5 space-y-1.5">
                      {r.drivers.length === 0 && <li className="text-[12px] text-muted">No individual feature stands out.</li>}
                      {r.drivers.slice(0, 4).map((d) => (
                        <li key={d.feature} className="flex items-center justify-between gap-3 text-[12px]">
                          <span className="min-w-0 truncate text-fg-2" title={d.label}>{d.label}</span>
                          <span className="flex shrink-0 items-center gap-2">
                            <span className="h-1 w-12 overflow-hidden rounded-full bg-surface-3"><span className="block h-full rounded-full bg-accent" style={{ width: `${Math.min(100, d.shap * 160)}%` }} /></span>
                            <span className="w-12 text-right font-mono text-[11px] tnum" style={{ color: d.z > 0 ? 'var(--warn)' : 'var(--info)' }}>{d.z > 0 ? '+' : ''}{d.z.toFixed(1)}σ</span>
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            )
          })}

          {verdict && (
            <div className="mt-auto flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-[12px]">
              {verdict === 'hit' ? <CheckCircle2 size={15} className="text-ok" /> : <XCircle size={15} className={verdict === 'top3' ? 'text-warn' : 'text-crit'} />}
              <span className="text-fg-2">
                Injected: <b className="text-fg">{COMPONENT_LABELS[truth!.target]}</b> · {FAULT_LABELS[truth!.type]}
              </span>
              <Badge tone={verdict === 'hit' ? 'ok' : verdict === 'top3' ? 'warn' : 'crit'} className="ml-auto">{verdict === 'hit' ? 'Top-1 correct' : verdict === 'top3' ? 'In top-3' : 'Missed'}</Badge>
            </div>
          )}
        </div>
      )}
    </Card>
  )
}
