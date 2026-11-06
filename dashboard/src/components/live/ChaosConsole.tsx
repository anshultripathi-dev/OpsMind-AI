import { useState } from 'react'
import { Cpu, MemoryStick, Database, Network, Power, Zap, Syringe, Eraser, Play, Flame, Film } from 'lucide-react'
import clsx from 'clsx'
import { Button, Card, CardHeader } from '@/components/ui'
import { useLive } from '@/lib/store'
import { actions } from '@/lib/actions'
import { COMPONENT_LABELS, FAULT_COLORS, FAULT_ORDER } from '@/lib/colors'
import type { FaultType } from '@/lib/types'

const ICONS: Record<FaultType, React.ComponentType<{ size?: number }>> = {
  cpu_saturation: Cpu,
  memory_leak: MemoryStick,
  db_slowdown: Database,
  network_latency: Network,
  service_crash: Power,
  cache_failure: Zap,
}

export function ChaosConsole() {
  const { config, status } = useLive()
  const [type, setType] = useState<FaultType>('db_slowdown')
  const [target, setTarget] = useState<string>('')
  const [intensity, setIntensity] = useState(0.8)
  const [duration, setDuration] = useState(300)
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null)
  const replay = status?.mode === 'replay'
  if (!config) return <div className="card skeleton h-[480px]" />

  const spec = config.faults[type]
  const tgt = target && spec.targets.includes(target as never) ? target : spec.primary ?? spec.targets[0]
  const run = (p: Promise<unknown>, text: string) =>
    p.then(() => setMsg({ ok: true, text })).catch((e) => setMsg({ ok: false, text: String(e.message) })).finally(() => setTimeout(() => setMsg(null), 3500))

  return (
    <Card className="flex h-full flex-col animate-fade-up">
      <CardHeader icon={<Syringe size={15} />} title="Chaos console" subtitle="Inject faults into the live simulator" />
      {replay ? (
        <div className="m-4 flex flex-1 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-line-strong p-6 text-center">
          <Film size={20} className="text-info" />
          <p className="text-[12.5px] text-fg-2">Replaying <b className="text-fg">{status?.source.run_id}</b>. Fault injection is available in live mode.</p>
          <Button variant="primary" size="sm" onClick={actions.reset}>Return to live simulator</Button>
        </div>
      ) : (
        <div className="flex flex-1 flex-col gap-3 px-4 pb-4">
          <div className="grid grid-cols-3 gap-1.5">
            {FAULT_ORDER.map((f) => {
              const Icon = ICONS[f]
              const on = f === type
              return (
                <button key={f} onClick={() => { setType(f); setTarget('') }} className={clsx('flex flex-col items-center gap-1.5 rounded-xl border px-2 py-2.5 text-center text-[11.5px] font-medium leading-tight transition-all', on ? 'border-transparent text-fg' : 'border-line bg-surface-2 text-fg-2 hover:border-line-strong')} style={on ? { background: `color-mix(in oklab, ${FAULT_COLORS[f]} 14%, var(--surface-2))`, boxShadow: `0 0 0 1.5px ${FAULT_COLORS[f]}` } : undefined}>
                  <span style={{ color: FAULT_COLORS[f] }}><Icon size={17} /></span>
                  {config.faults[f].label}
                </button>
              )
            })}
          </div>

          <div className="card-flat space-y-3 p-3">
            <div className="text-[11.5px] text-muted">Tool: <span className="mono text-fg-2">{spec.tool}</span></div>
            <div className="grid grid-cols-2 gap-3">
              <label className="space-y-1">
                <span className="eyebrow">Target</span>
                <select value={tgt} onChange={(e) => setTarget(e.target.value)} className="h-8 w-full rounded-lg border border-line-strong bg-surface px-2 text-[12.5px] text-fg outline-none focus:border-accent">
                  {spec.targets.map((t) => <option key={t} value={t}>{COMPONENT_LABELS[t]}</option>)}
                </select>
              </label>
              <label className="space-y-1">
                <span className="eyebrow">Duration</span>
                <select value={duration} onChange={(e) => setDuration(Number(e.target.value))} className="h-8 w-full rounded-lg border border-line-strong bg-surface px-2 text-[12.5px] text-fg outline-none focus:border-accent">
                  {[180, 300, 420, 600].map((d) => <option key={d} value={d}>{d / 60} min</option>)}
                </select>
              </label>
            </div>
            <label className="block space-y-1">
              <span className="flex items-center justify-between"><span className="eyebrow">Intensity</span><span className="font-mono text-[11.5px] tnum text-fg-2">{intensity.toFixed(2)}</span></span>
              <input type="range" min={0.3} max={1} step={0.05} value={intensity} onChange={(e) => setIntensity(Number(e.target.value))} className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-surface-3 accent-[var(--accent)]" />
            </label>
            <Button variant="primary" className="w-full" icon={<Flame size={14} />} onClick={() => run(actions.inject(type, tgt, duration, intensity), `${spec.label} injected on ${COMPONENT_LABELS[tgt as never]}`)}>
              Inject {spec.label.toLowerCase()}
            </Button>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <Button icon={<Play size={13} />} onClick={() => run(actions.spike(4, 150), 'Healthy 4× traffic burst started')}>Healthy traffic burst</Button>
            <Button variant="danger" icon={<Eraser size={13} />} onClick={() => run(actions.clear(), 'Faults cleared')}>Clear all</Button>
          </div>

          <div className="space-y-1.5">
            <div className="eyebrow">Scenarios</div>
            {config.scenarios.map((s) => (
              <button key={s.id} onClick={() => run(actions.scenario(s.id), `Scenario “${s.title}” started`)} className="group flex w-full items-start gap-2.5 rounded-lg border border-line bg-surface-2/60 px-3 py-2 text-left transition-colors hover:border-accent/50 hover:bg-surface-2">
                <Play size={12} className="mt-[3px] shrink-0 text-muted transition-colors group-hover:text-accent" />
                <span className="min-w-0">
                  <span className="block text-[12.5px] font-medium">{s.title}</span>
                  <span className="block truncate text-[11.5px] text-muted">{s.description}</span>
                </span>
              </button>
            ))}
          </div>

          <div className={clsx('mt-auto min-h-[20px] text-[12px] transition-opacity', msg ? 'opacity-100' : 'opacity-0', msg?.ok ? 'text-ok' : 'text-crit')}>{msg?.text}</div>
        </div>
      )}
    </Card>
  )
}
