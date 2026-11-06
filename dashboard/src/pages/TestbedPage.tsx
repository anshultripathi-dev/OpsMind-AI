import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Boxes, Database, FileText, Syringe } from 'lucide-react'
import { Badge, Card, CardHeader, ChartTooltip, Skeleton } from '@/components/ui'
import { useDataset } from '@/lib/useBench'
import { useLive } from '@/lib/store'
import { COMPONENT_LABELS, FAULT_COLORS, FAULT_LABELS, FAULT_ORDER } from '@/lib/colors'

/* eslint-disable @typescript-eslint/no-explicit-any */
const axis = { tick: { fill: 'var(--muted)', fontSize: 10.5 }, axisLine: false, tickLine: false } as const

function Box({ x, y, w = 138, h = 50, title, sub, tone = 'var(--border-strong)' }: { x: number; y: number; w?: number; h?: number; title: string; sub: string; tone?: string }) {
  return (
    <g transform={`translate(${x},${y})`}>
      <rect width={w} height={h} rx={10} fill="var(--surface-2)" stroke={tone} />
      <text x={w / 2} y={h / 2 - 3} textAnchor="middle" fontSize={11.5} fontWeight={600} fill="var(--text)">{title}</text>
      <text x={w / 2} y={h / 2 + 12} textAnchor="middle" fontSize={9.5} fill="var(--muted)">{sub}</text>
    </g>
  )
}
function Arrow({ d }: { d: string }) {
  return <path d={d} fill="none" stroke="var(--border-strong)" strokeWidth={1.4} markerEnd="url(#ah)" />
}

function Architecture() {
  return (
    <svg viewBox="0 0 1040 250" className="w-full" role="img" aria-label="System architecture">
      <defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="var(--muted)" /></marker></defs>
      <Box x={4} y={14} title="Microservice testbed" sub="Docker Compose · 4 services + DB + cache" tone="var(--info)" w={170} />
      <Box x={4} y={96} title="Fault injector + Locust" sub="stress-ng · Pumba · docker kill" tone="var(--crit)" w={170} />
      <Box x={4} y={178} title="Telemetry" sub="Prometheus · cAdvisor · JSON logs" w={170} />
      <Arrow d="M 89 64 L 89 96" /><Arrow d="M 89 146 L 89 178" />
      <Box x={226} y={96} title="Preprocessing" sub="windows · rolling stats · Drain" w={150} />
      <Arrow d="M 174 203 C 200 203, 200 121, 226 121" />
      {[['Anomaly detection', 'One-Class SVM · IF · PCA'], ['Classification', 'LR · SVM · RF · XGB · stack'], ['Clustering', 'K-means · Ward · DBSCAN'], ['Early-warning LSTM', '5-min sequence → P(incident)']].map(([t, s], i) => (
        <g key={t}><Box x={428} y={4 + i * 60} title={t} sub={s} w={190} h={48} tone="var(--accent)" /><Arrow d={`M 376 121 C 400 121, 404 ${28 + i * 60}, 428 ${28 + i * 60}`} /></g>
      ))}
      <Box x={670} y={96} title="Root-cause ranking" sub="SHAP + dependency graph" w={150} tone="var(--accent)" />
      {[0, 1, 2, 3].map((i) => <Arrow key={i} d={`M 618 ${28 + i * 60} C 644 ${28 + i * 60}, 644 121, 670 121`} />)}
      <Box x={868} y={40} title="FastAPI service" sub="REST · WebSocket · replay" w={168} />
      <Box x={868} y={150} title="React dashboard" sub="live ops · incidents · model lab" w={168} tone="var(--ok)" />
      <Arrow d="M 820 121 C 846 121, 846 65, 868 65" /><Arrow d="M 952 90 L 952 150" />
    </svg>
  )
}

export function TestbedPage() {
  const ds = useDataset()
  const config = useLive((s) => s.config)
  if (!ds || !config) return <div className="space-y-4">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-60" />)}</div>
  const s = ds.stats
  const byFault = FAULT_ORDER.map((f) => ({ f, episodes: s.episodes_by_fault[f] ?? 0, impacted: s.impacted_by_fault[f] ?? 0 }))
  const kpis = [
    ['Experiment runs', s.runs, `${s.total_hours} h of 30-min runs`],
    ['Fault episodes', s.fault_episodes, `${s.impacted_episodes} breached the user SLO`],
    ['Normal operation', `${s.normal_hours} h`, `${s.normal_runs} runs incl. healthy spikes`],
    ['Labelled windows', s.windows.toLocaleString(), `30 s windows, 10 s stride`],
    ['Features', s.n_features, `${s.n_metric_features} metric + ${s.n_log_features} log`],
    ['Log lines parsed', s.log_lines.toLocaleString(), `${s.n_templates} Drain templates`],
  ]
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader icon={<Boxes size={15} />} title="System architecture" subtitle="Data flows one way: testbed → telemetry → features → four ML modules → root cause → API → dashboard" />
        <div className="px-4 pb-4"><Architecture /></div>
      </Card>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3 xl:grid-cols-6">
        {kpis.map(([l, v, sub]) => (
          <Card key={String(l)} className="p-4"><div className="eyebrow">{l}</div><div className="mt-1 text-[24px] font-semibold leading-none tracking-tight tnum">{v}</div><div className="mt-1.5 text-[11.5px] text-muted">{sub}</div></Card>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-12">
        <Card className="xl:col-span-7">
          <CardHeader icon={<Syringe size={15} />} title="Fault catalogue" subtitle="Labels come from the injector log only — no manual labelling" />
          <div className="overflow-x-auto px-2 pb-3">
            <table className="w-full min-w-[620px] text-left text-[12.5px]">
              <thead><tr className="eyebrow border-b border-line"><th className="px-3 py-2">Fault</th><th>Injected with</th><th>Targets</th><th className="pl-4 text-right">Episodes</th><th className="pl-4 pr-3 text-right">Impacting</th></tr></thead>
              <tbody>
                {byFault.map(({ f, episodes, impacted }) => (
                  <tr key={f} className="border-b border-line/60 last:border-0">
                    <td className="px-3 py-2.5 font-medium"><span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: FAULT_COLORS[f] }} />{FAULT_LABELS[f]}</td>
                    <td className="mono text-[11.5px] text-fg-2">{config.faults[f].tool}</td>
                    <td className="text-fg-2">{config.faults[f].targets.map((t) => COMPONENT_LABELS[t]).join(', ')}</td>
                    <td className="text-right tnum">{episodes}</td><td className="pr-3 text-right tnum text-fg-2">{impacted}</td>
                  </tr>
                ))}
                <tr><td className="px-3 py-2.5 font-medium"><span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: FAULT_COLORS.normal }} />Traffic spike <Badge tone="ok">labelled normal</Badge></td><td className="mono text-[11.5px] text-fg-2">Locust burst on the gateway</td><td className="text-fg-2">API Gateway</td><td className="text-right text-muted" colSpan={2}>healthy by design</td></tr>
              </tbody>
            </table>
          </div>
        </Card>
        <Card className="xl:col-span-5">
          <CardHeader icon={<Database size={15} />} title="Episodes per fault & split" subtitle={`Chronological per fault type · windows: ${Object.entries(s.windows_by_split).map(([k, v]) => `${k} ${(v as number).toLocaleString()}`).join(' · ')}`} />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={byFault.map((d) => ({ name: FAULT_LABELS[d.f].split(' ')[0], Episodes: d.episodes, Impacting: d.impacted, f: d.f }))} margin={{ top: 8, right: 10, left: -18, bottom: 0 }} barGap={3}>
                <CartesianGrid vertical={false} /><XAxis dataKey="name" {...axis} /><YAxis {...axis} />
                <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTooltip />} />
                <Bar dataKey="Episodes" radius={[4, 4, 0, 0]} isAnimationActive={false}>{byFault.map((d) => <Cell key={d.f} fill={FAULT_COLORS[d.f]} fillOpacity={0.55} />)}</Bar>
                <Bar dataKey="Impacting" radius={[4, 4, 0, 0]} isAnimationActive={false}>{byFault.map((d) => <Cell key={d.f} fill={FAULT_COLORS[d.f]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="px-3 text-[11.5px] text-muted">Split by whole experiment run, never by random window: {Object.entries(s.runs_by_split).map(([k, v]) => `${k} ${v}`).join(' · ')} runs.</p>
          </div>
        </Card>
      </div>

      {ds.loghub && (
        <Card className="p-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="eyebrow">Public benchmark · log parsing</div>
              <div className="mt-1 text-[15px] font-semibold">Drain on {ds.loghub.dataset}</div>
              <div className="mt-0.5 text-[12px] text-muted">{ds.loghub.lines.toLocaleString()} lines · {ds.loghub.ground_truth_templates} expert templates · Drain found {ds.loghub.drain_templates} · {ds.loghub.published_reference}</div>
            </div>
            <div className="text-right"><div className="eyebrow">Grouping accuracy</div><div className="text-[28px] font-semibold leading-none tracking-tight tnum text-ok">{(ds.loghub.grouping_accuracy * 100).toFixed(2)}%</div></div>
          </div>
        </Card>
      )}

      <Card>
        <CardHeader icon={<FileText size={15} />} title="Log event templates discovered by Drain" subtitle="Variable parts (ids, durations, counts) are masked; each template becomes a count feature per service" />
        <div className="grid gap-x-6 gap-y-0.5 px-4 pb-4 font-mono text-[11.5px] md:grid-cols-2">
          {Object.entries(ds.templates).map(([id, t]) => (
            <div key={id} className="flex gap-3 rounded px-2 py-[3px] hover:bg-surface-2"><span className="w-8 shrink-0 text-muted">T{id}</span><span className="truncate text-fg-2" title={String(t)}>{String(t).replace(/<NUM>/g, '#').replace(/<\*>/g, '*')}</span></div>
          ))}
        </div>
      </Card>
    </div>
  )
}
