import { useMemo, useState } from 'react'
import { BellRing, CheckCircle2, Eraser, Radar, ScrollText, ShieldAlert, Siren, Sparkles, Syringe, TriangleAlert, ListTree } from 'lucide-react'
import clsx from 'clsx'
import { Badge, Card, CardHeader, Empty, Segmented } from '@/components/ui'
import { useLive } from '@/lib/store'
import { COMPONENT_COLORS, COMPONENT_LABELS, FAULT_COLORS, FAULT_LABELS } from '@/lib/colors'
import { fmtClock, fmtPct } from '@/lib/format'
import type { EventRec } from '@/lib/types'

const EV: Record<EventRec['kind'], { Icon: React.ComponentType<{ size?: number }>; tone: string }> = {
  early_warning: { Icon: BellRing, tone: 'var(--warn)' },
  detected: { Icon: Radar, tone: 'var(--accent)' },
  threshold_alert: { Icon: Siren, tone: 'var(--muted)' },
  impact: { Icon: TriangleAlert, tone: 'var(--crit)' },
  resolved: { Icon: CheckCircle2, tone: 'var(--ok)' },
  injected: { Icon: Syringe, tone: 'var(--info)' },
  cleared: { Icon: Eraser, tone: 'var(--muted)' },
  novel_pattern: { Icon: Sparkles, tone: 'var(--warn)' },
}

export function EventFeed() {
  const events = useLive((s) => s.events)
  const list = useMemo(() => [...events].reverse().slice(0, 40), [events])
  return (
    <Card className="flex h-full flex-col animate-fade-up">
      <CardHeader icon={<ShieldAlert size={15} />} title="Alert & event feed" subtitle="What OpsMind and the static rules did, in order" actions={<Badge tone="neutral">{events.length}</Badge>} />
      {list.length === 0 ? (
        <Empty icon={<ScrollText size={18} />} title="No events yet" hint="Alerts, detections and operator actions appear here." className="flex-1" />
      ) : (
        <ol className="max-h-[430px] flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
          {list.map((e, i) => {
            const { Icon, tone } = EV[e.kind] ?? EV.cleared
            return (
              <li key={`${e.t}-${e.kind}-${i}`} className={clsx('flex gap-3 rounded-lg px-2.5 py-2 transition-colors hover:bg-surface-2', i === 0 && 'animate-fade-up')}>
                <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-lg" style={{ background: `color-mix(in oklab, ${tone} 14%, transparent)`, color: tone }}>
                  <Icon size={13} />
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[12.5px] font-medium">{e.title}</span>
                    <span className="shrink-0 font-mono text-[10.5px] tnum text-muted">{fmtClock(e.t)}</span>
                  </div>
                  <div className="truncate text-[11.5px] text-muted">{e.detail}</div>
                </div>
              </li>
            )
          })}
        </ol>
      )}
    </Card>
  )
}

const LEVEL_TONE: Record<string, 'warn' | 'crit' | 'neutral'> = { WARN: 'warn', ERROR: 'crit', FATAL: 'crit' }

export function LogStream() {
  const ticks = useLive((s) => s.ticks)
  const [level, setLevel] = useState<'ALL' | 'ERROR'>('ALL')
  const rows = useMemo(() => {
    const out: { key: string; t: number; service: keyof typeof COMPONENT_COLORS; level: string; msg: string; tid: number }[] = []
    for (const k of ticks.slice(-40)) k.logs.forEach((l, j) => out.push({ key: `${k.i}-${j}`, t: l.t, service: l.service, level: l.level, msg: l.msg, tid: l.template_id }))
    return out.filter((r) => level === 'ALL' || r.level !== 'WARN').slice(-60).reverse()
  }, [ticks, level])
  return (
    <Card className="flex h-full flex-col animate-fade-up">
      <CardHeader icon={<ListTree size={15} />} title="Log stream · Drain templates" subtitle="Warnings and errors parsed into event templates (INFO lines are counted, not shown)" actions={<Segmented size="sm" value={level} onChange={setLevel} options={[{ value: 'ALL', label: 'Warn + error' }, { value: 'ERROR', label: 'Errors' }]} />} />
      {rows.length === 0 ? (
        <Empty icon={<ScrollText size={18} />} title="Quiet logs" hint="No warnings or errors in the recent windows." className="flex-1" />
      ) : (
        <div className="max-h-[300px] flex-1 overflow-y-auto px-2 pb-3 font-mono text-[11.5px]">
          {rows.map((r) => (
            <div key={r.key} className="flex items-center gap-2.5 rounded px-2 py-[3px] hover:bg-surface-2">
              <span className="w-[58px] shrink-0 text-muted tnum">{fmtClock(r.t)}</span>
              <Badge tone={LEVEL_TONE[r.level] ?? 'neutral'} className="w-[46px] shrink-0 justify-center !px-1 !text-[10px]">{r.level}</Badge>
              <span className="flex w-[92px] shrink-0 items-center gap-1.5 text-fg-2"><span className="h-1.5 w-1.5 rounded-full" style={{ background: COMPONENT_COLORS[r.service] }} />{COMPONENT_LABELS[r.service]}</span>
              <span className="min-w-0 flex-1 truncate text-fg" title={r.msg}>{r.msg}</span>
              <span className="shrink-0 rounded bg-surface-3 px-1.5 text-[10px] text-muted">T{r.tid}</span>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

export function ClassifierCard() {
  const last = useLive((s) => s.ticks[s.ticks.length - 1])
  if (!last) return <div className="card skeleton h-[300px]" />
  const probs = Object.entries(last.classifier.probs).sort((a, b) => b[1] - a[1])
  const p = last.pattern
  return (
    <Card className="flex h-full flex-col animate-fade-up">
      <CardHeader icon={<Sparkles size={15} />} title="Incident type & failure pattern" subtitle="Ensemble classifier + unsupervised clustering" />
      <div className="space-y-2 px-4 pb-3">
        {probs.slice(0, 4).map(([k, v]) => (
          <div key={k}>
            <div className="mb-1 flex items-center justify-between text-[12px]">
              <span className={clsx(k === last.classifier.label ? 'font-semibold text-fg' : 'text-fg-2')}>{FAULT_LABELS[k] ?? k}</span>
              <span className="tnum text-muted">{fmtPct(v)}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
              <div className="h-full rounded-full transition-all duration-500" style={{ width: `${Math.max(v * 100, 1)}%`, background: FAULT_COLORS[k] }} />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-auto border-t border-line px-4 py-3">
        <div className="mb-1 flex items-center justify-between">
          <span className="eyebrow">Nearest known pattern</span>
          {p.novel ? <Badge tone="warn">Unseen</Badge> : <Badge tone="neutral">dist {p.distance}</Badge>}
        </div>
        <div className="text-[13px] font-semibold">{p.name}{p.dominant && <span className="ml-2 font-normal text-muted">≈ {FAULT_LABELS[p.dominant] ?? p.dominant} ({fmtPct(p.purity ?? 0)} of history)</span>}</div>
        <div className="mt-0.5 text-[11.5px] text-muted">{p.signature}</div>
      </div>
    </Card>
  )
}
