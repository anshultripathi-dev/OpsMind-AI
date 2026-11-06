import { Pause, Play, RotateCcw, Radio, Film, Gauge as GaugeIcon } from 'lucide-react'
import { useLocation } from 'react-router-dom'
import { Badge, Button, Segmented } from '@/components/ui'
import { useLive } from '@/lib/store'
import { actions } from '@/lib/actions'
import { fmtClock } from '@/lib/format'

const TITLES: Record<string, { title: string; sub: string }> = {
  '/': { title: 'Live operations', sub: 'Streaming telemetry through the full ML pipeline' },
  '/incidents': { title: 'Incidents', sub: 'Detected incidents, warning lead time and replayable test episodes' },
  '/lab': { title: 'Model lab', sub: 'Every module compared against its baseline on held-out experiment runs' },
  '/testbed': { title: 'Testbed & data', sub: 'Microservice testbed, fault catalogue and the labelled dataset' },
}

const SPEEDS = [
  { value: 0.5, label: '5×' },
  { value: 1, label: '10×' },
  { value: 2, label: '20×' },
  { value: 5, label: '50×' },
]

export function Topbar() {
  const { pathname } = useLocation()
  const { status, ticks } = useLive()
  const meta = TITLES[pathname] ?? TITLES['/']
  const last = ticks[ticks.length - 1]
  const replay = status?.mode === 'replay'
  return (
    <header className="sticky top-0 z-30 flex items-center justify-between gap-4 border-b border-line bg-bg/80 px-5 py-3 backdrop-blur-xl">
      <div className="min-w-0">
        <h1 className="truncate text-[17px] font-semibold tracking-tight">{meta.title}</h1>
        <p className="hidden truncate text-[12px] text-muted sm:block">{meta.sub}</p>
      </div>
      <div className="flex items-center gap-2.5">
        {status && (
          <Badge tone={replay ? 'info' : 'ok'} dot className="hidden sm:inline-flex">
            {replay ? <><Film size={11} /> Replay · {status.source.run_id}{status.models === 'real' ? ' · real-trained models' : ''}</> : <><Radio size={11} /> Live simulator</>}
          </Badge>
        )}
        <div className="hidden items-center gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1 font-mono text-[12px] tnum text-fg-2 md:flex" title="Simulated wall clock">
          {last ? fmtClock(last.t) : '--:--:--'}
          <span className="text-muted">sim</span>
        </div>
        <div className="hidden items-center gap-1.5 md:flex">
          <GaugeIcon size={13} className="text-muted" />
          <Segmented size="sm" value={status?.speed ?? 1} onChange={(v) => actions.speed(v)} options={SPEEDS} />
        </div>
        {replay && status?.replay_done && (
          <Button size="sm" variant="primary" onClick={actions.reset}>Replay finished · back to live</Button>
        )}
        {status?.paused ? (
          <Button size="sm" variant="primary" icon={<Play size={13} />} onClick={actions.resume}>Resume</Button>
        ) : (
          <Button size="sm" icon={<Pause size={13} />} onClick={actions.pause}>Pause</Button>
        )}
        <Button size="sm" variant="ghost" icon={<RotateCcw size={13} />} onClick={actions.reset} title="Restart live simulator">
          <span className="hidden sm:inline">Live</span>
        </Button>
      </div>
    </header>
  )
}
