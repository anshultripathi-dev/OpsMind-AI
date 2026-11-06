import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, CheckCircle2, Film, MinusCircle, ShieldAlert, XCircle, Clock } from 'lucide-react'
import { Badge, Button, Card, CardHeader, Empty, Segmented, Skeleton } from '@/components/ui'
import { useLive } from '@/lib/store'
import { useDataset } from '@/lib/useBench'
import { actions } from '@/lib/actions'
import { COMPONENT_LABELS, FAULT_COLORS, FAULT_LABELS } from '@/lib/colors'
import { fmtClock, fmtDur } from '@/lib/format'

/* eslint-disable @typescript-eslint/no-explicit-any */
function Lead({ ep }: { ep: any }) {
  if (!ep) return <span className="text-muted">n/a</span>
  if (ep.status === 'warned') return <Badge tone="ok"><Clock size={11} />+{fmtDur(ep.lead)}</Badge>
  if (ep.status === 'late') return <Badge tone="warn">{fmtDur(ep.lead)} late</Badge>
  return <Badge tone="neutral"><MinusCircle size={11} />missed</Badge>
}

function FaultChip({ f }: { f: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ background: FAULT_COLORS[f] }} />
      {FAULT_LABELS[f] ?? f}
    </span>
  )
}

export function IncidentsPage() {
  const [tab, setTab] = useState<'session' | 'episodes'>('episodes')
  const ticks = useLive((s) => s.ticks)
  const last = ticks[ticks.length - 1]
  const session = useMemo(() => (last ? [...(last.incident.current ? [last.incident.current] : []), ...last.incident.history] : []), [last])
  const ds = useDataset()
  const nav = useNavigate()
  const eps: any[] = ds?.episodes ?? []
  const impacted = eps.filter((e) => e.impact_start !== null)
  const stat = (k: 'lstm' | 'threshold') => impacted.filter((e) => e[k]?.status === 'warned').length

  const replay = (id: string) => actions.replay(id).then(() => nav('/'))

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          { label: 'Held-out fault episodes', value: eps.length || '—', sub: `${impacted.length} caused user impact` },
          { label: 'Warned before impact · OpsMind', value: impacted.length ? `${stat('lstm')}/${impacted.length}` : '—', sub: 'LSTM early-warning', tone: 'var(--accent)' },
          { label: 'Warned before impact · static rules', value: impacted.length ? `${stat('threshold')}/${impacted.length}` : '—', sub: 'Prometheus-style thresholds' },
          { label: 'Root cause in top-1', value: eps.length ? `${eps.filter((e) => e.rca_top3[0] === e.target).length}/${eps.length}` : '—', sub: `${eps.filter((e) => e.rca_top3.includes(e.target)).length}/${eps.length} in top-3` },
        ].map((k) => (
          <Card key={k.label} className="p-4">
            <div className="eyebrow">{k.label}</div>
            <div className="mt-1 text-[28px] font-semibold leading-none tracking-tight tnum" style={{ color: k.tone }}>{k.value}</div>
            <div className="mt-1.5 text-[11.5px] text-muted">{k.sub}</div>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader icon={<ShieldAlert size={15} />} title="Incident log" subtitle={tab === 'session' ? 'Incidents detected by the live pipeline during this session' : 'Test-split experiment runs the models never saw — replay any of them through the live pipeline'} actions={<Segmented value={tab} onChange={setTab} options={[{ value: 'episodes', label: 'Test episodes' }, { value: 'session', label: `This session (${session.length})` }]} />} />
        <div className="overflow-x-auto px-2 pb-3">
          {tab === 'session' ? (
            session.length === 0 ? (
              <Empty icon={<ShieldAlert size={18} />} title="No incidents in this session" hint="Inject a fault from Live operations, or replay a test episode." />
            ) : (
              <table className="w-full min-w-[820px] text-left text-[12.5px]">
                <thead><tr className="eyebrow border-b border-line"><th className="px-3 py-2">Incident</th><th>Type</th><th>Root cause (top-3)</th><th>Detected</th><th>Duration</th><th>OpsMind warning</th><th>Static alert</th><th>Status</th></tr></thead>
                <tbody>
                  {session.map((i) => (
                    <tr key={i.id} className="border-b border-line/60 last:border-0 hover:bg-surface-2">
                      <td className="px-3 py-2.5 font-mono font-medium">{i.id}</td>
                      <td>{i.type ? <FaultChip f={i.type} /> : '—'}</td>
                      <td>{i.rca?.map((r, k) => <span key={r.component} className={k === 0 ? 'mr-2 font-semibold' : 'mr-2 text-muted'}>{COMPONENT_LABELS[r.component]}</span>)}</td>
                      <td className="font-mono tnum text-fg-2">{fmtClock(i.detected_at)}</td>
                      <td className="tnum">{fmtDur((i.resolved_at ?? last!.t) - i.started_at)}</td>
                      <td>{i.lead_s !== null ? <Badge tone="ok"><Clock size={11} />+{fmtDur(i.lead_s)}</Badge> : <span className="text-muted">{i.warning_at ? 'no impact' : 'none'}</span>}</td>
                      <td>{i.baseline_lead_s !== null ? <Badge tone={i.baseline_lead_s > 0 ? 'neutral' : 'warn'}>{i.baseline_lead_s > 0 ? `+${fmtDur(i.baseline_lead_s)}` : `${fmtDur(i.baseline_lead_s)} late`}</Badge> : <span className="text-muted">none</span>}</td>
                      <td><Badge tone={i.status === 'open' ? 'crit' : 'ok'} dot>{i.status}</Badge></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )
          ) : !ds ? (
            <div className="space-y-2 p-3">{[0, 1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-9" />)}</div>
          ) : (
            <table className="w-full min-w-[980px] text-left text-[12.5px]">
              <thead><tr className="eyebrow border-b border-line"><th className="px-3 py-2">Run</th><th>Injected fault</th><th>Target</th><th>Intensity</th><th>Classified as</th><th>Root cause top-3</th><th>OpsMind (LSTM)</th><th>Static rules</th><th /></tr></thead>
              <tbody>
                {eps.map((e) => {
                  const ok = e.predicted_type === e.fault_type
                  return (
                    <tr key={e.run_id} className="border-b border-line/60 last:border-0 hover:bg-surface-2">
                      <td className="px-3 py-2 font-mono text-fg-2">{e.run_id}</td>
                      <td><FaultChip f={e.fault_type} /></td>
                      <td>{COMPONENT_LABELS[e.target as keyof typeof COMPONENT_LABELS]}</td>
                      <td className="tnum text-fg-2">{e.intensity.toFixed(2)}</td>
                      <td><span className="inline-flex items-center gap-1.5">{ok ? <CheckCircle2 size={13} className="text-ok" /> : <XCircle size={13} className="text-crit" />}{FAULT_LABELS[e.predicted_type]}</span></td>
                      <td>{e.rca_top3.map((c: string, k: number) => <span key={c} className={`mr-2 ${c === e.target ? 'font-semibold text-ok' : k === 0 ? 'font-semibold' : 'text-muted'}`}>{COMPONENT_LABELS[c as keyof typeof COMPONENT_LABELS]}</span>)}</td>
                      <td>{e.impact_start === null ? <span className="text-muted">no impact</span> : <Lead ep={e.lstm} />}</td>
                      <td>{e.impact_start === null ? <span className="text-muted">no impact</span> : <Lead ep={e.threshold} />}</td>
                      <td className="pr-3 text-right"><Button size="sm" variant="ghost" icon={<Film size={12} />} onClick={() => replay(e.run_id)}>Replay <ArrowRight size={11} /></Button></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </Card>
    </div>
  )
}
