import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Film } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { Badge, Button, Card, CardHeader, ChartTooltip, Segmented, Skeleton } from '@/components/ui'
import { get } from '@/lib/api'
import { actions } from '@/lib/actions'
import { Criterion, Section, f3, heat, pct } from '@/components/lab/parts'
import { useBench, useSources } from '@/lib/useBench'
import { CLUSTER_COLORS, COMPONENT_LABELS, FAULT_COLORS, FAULT_LABELS, FAULT_ORDER } from '@/lib/colors'

/* eslint-disable @typescript-eslint/no-explicit-any */
const axis = { tick: { fill: 'var(--muted)', fontSize: 10.5 }, axisLine: false, tickLine: false } as const
const ACC = 'var(--accent)'
const GREY = 'var(--border-strong)'

function clipDomain(points: any[], key: 'x' | 'y'): [number, number] {
  const v = points.map((p) => p[key]).sort((a, b) => a - b)
  const lo = v[Math.floor(v.length * 0.01)]
  const hi = v[Math.floor(v.length * 0.99)]
  const pad = (hi - lo) * 0.12
  return [Math.floor(lo - pad), Math.ceil(hi + pad)]
}

const NAV = [
  ['summary', 'Summary'], ['dimred', 'PCA'], ['anomaly', 'Anomaly'], ['classification', 'Classification'],
  ['clustering', 'Clustering'], ['warning', 'Early warning'], ['rca', 'Root cause'],
]

export function ModelLabPage() {
  const [source, setSource] = useState<'simulator' | 'real'>('simulator')
  const sources = useSources()
  const b = useBench(source)
  if (!b) return <div className="space-y-4">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-60" />)}</div>

  const an = b.anomaly.models as any[]
  const base = an.find((m) => m.key === 'threshold')
  const ml = an.filter((m) => m.key !== 'threshold')
  const best = ml.reduce((a, c) => (c.f1 > a.f1 ? c : a))
  const cl = b.classification
  const bestSingle = cl.models.filter((m: any) => m.family === 'single').reduce((a: any, c: any) => (c.macro_f1 > a.macro_f1 ? c : a))
  const bestEns = cl.models.filter((m: any) => m.family === 'ensemble').reduce((a: any, c: any) => (c.macro_f1 > a.macro_f1 ? c : a))
  const ew = b.earlywarning.models as any[]
  const ewBase = ew.find((m) => m.model.startsWith('Static'))
  const lstm = ew.find((m) => m.model === 'LSTM')
  const rca = b.rootcause.methods as any[]
  const rcaFull = rca.find((m) => m.method.includes('graph'))
  const km = b.clustering.algorithms.find((a: any) => a.model === 'K-means')
  const rnd = b.clustering.algorithms.find((a: any) => a.model === 'Random assignment')

  return (
    <div className="space-y-10">
      <div className="sticky top-[61px] z-20 -mx-5 flex gap-1 overflow-x-auto border-b border-line bg-bg/85 px-5 py-2 backdrop-blur-xl">
        {[...NAV, ...(source === 'simulator' ? [['real', 'Real testbed']] : [])].map(([id, label]) => (
          <a key={id} href={`#${id}`} onClick={(e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }) }} className="whitespace-nowrap rounded-lg px-3 py-1.5 text-[12.5px] font-medium text-fg-2 transition-colors hover:bg-surface-3 hover:text-fg">{label}</a>
        ))}
      </div>

      {sources.real && (
        <div className="flex items-center justify-between gap-4 rounded-xl border border-line bg-surface px-4 py-3">
          <div className="text-[12.5px] text-fg-2">{source === 'real' ? <>Models <b className="text-fg">trained and tested on real Docker-testbed runs</b> ({b.meta?.n_features} features). Few runs, so differences are noisy.</> : <>Models trained and tested on <b className="text-fg">simulated runs</b>. Switch to see the same pipeline trained on real telemetry.</>}</div>
          <Segmented value={source} onChange={setSource} options={[{ value: 'simulator', label: 'Simulator data' }, { value: 'real', label: 'Real Docker data' }]} />
        </div>
      )}
      <Section id="summary" eyebrow="Success criteria" title="Does it work? Measured on unseen experiment runs" desc="All numbers are on the test split: whole fault episodes the models never saw, split chronologically per fault type. Every module is compared against the baseline named in the proposal.">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Criterion ok={best.f1 > base.f1} title="Anomaly F1 vs thresholds" value={`${f3(best.f1)} vs ${f3(base.f1)}`} detail={`${best.model} beats static alert rules on window-level F1.`} />
          <Criterion ok={best.false_alarms_per_hour < base.false_alarms_per_hour} title="False alarms / hour" value={`${best.false_alarms_per_hour.toFixed(2)} vs ${base.false_alarms_per_hour.toFixed(2)}`} detail="Sustained alerts on healthy periods, including benign traffic spikes and backups." />
          <Criterion ok={lstm.recall > ewBase.recall} title="Warned before user impact" value={`${pct(lstm.recall, 0)} vs ${pct(ewBase.recall, 0)}`} detail={`LSTM early warning vs static rules · mean lead ${Math.round(lstm.mean_lead_s)} s. Sudden faults (crash, cache) are inherently unpredictable.`} />
          <Criterion ok={rcaFull.episode.top3 >= 0.9} title="Root cause in top-3" value={pct(rcaFull.episode.top3, 0)} detail={`Top-1 ${pct(rcaFull.episode.top1, 0)} vs ${pct(rca[0].episode.top1, 0)} for “highest raw metric”.`} />
        </div>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Criterion ok={null} title="Best single classifier" value={f3(bestSingle.macro_f1)} detail={`${bestSingle.model} · macro-F1 on fault windows`} />
          <Criterion ok={null} title="Best ensemble" value={f3(bestEns.macro_f1)} detail={`${bestEns.model} · ${bestEns.macro_f1 > bestSingle.macro_f1 ? 'edges out' : 'matches'} the best single model`} />
          <Criterion ok={km.ari > rnd.ari + 0.3} title="Clustering ARI" value={f3(km.ari)} detail={`K-means (k=${km.n_clusters}) vs ${f3(rnd.ari)} for random assignment · no labels used`} />
          <Criterion ok={null} title="Detector + classifier" value={`${b.classification.pipeline.detector_plus_classifier.false_alarms_per_hour.toFixed(2)} FA/h`} detail={`Letting the classifier veto benign anomalies cuts false alarms from ${b.classification.pipeline.detector_only.false_alarms_per_hour.toFixed(2)}/h.`} />
        </div>
      </Section>

      <DimRed d={b.dimred} />
      <Anomaly an={an} />
      <Classification cl={cl} />
      <Clustering c={b.clustering} />
      <Warning ew={ew} hist={b.earlywarning.history} histG={b.earlywarning.history_gru} />
      <RootCause methods={rca} />
      {source === 'simulator' && <RealCheck />}
    </div>
  )
}

/* ───────────────────────────── PCA ───────────────────────────── */
function DimRed({ d }: { d: any }) {
  const cum = d.report.cumulative_variance.map((v: number, i: number) => ({ k: i + 1, v }))
  const groups = useMemo(() => {
    const g: Record<string, any[]> = {}
    d.map.forEach((p: any) => (g[p.label] ??= []).push(p))
    return g
  }, [d])
  return (
    <Section id="dimred" eyebrow="Module 2 · Dimensionality reduction" title="PCA compresses 300 features into a handful of components" desc={<>{d.report.n_input_features} engineered features reduce to <b>{d.chosen_components}</b> components at 95 % variance. The first two components already separate fault families into a 2-D map of system states.</>}>
      <div className="grid gap-4 xl:grid-cols-12">
        <Card className="xl:col-span-7">
          <CardHeader title="System-state map (test windows)" subtitle={`PC1 ${pct(d.explained_2d[0], 0)} · PC2 ${pct(d.explained_2d[1], 0)} of variance`} />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" name="PC1" domain={clipDomain(d.map, 'x')} allowDataOverflow {...axis} /><YAxis type="number" dataKey="y" name="PC2" domain={clipDomain(d.map, 'y')} allowDataOverflow {...axis} /><ZAxis range={[14, 14]} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<ChartTooltip labelFormatter={() => 'State'} valueFormatter={(v: number) => v.toFixed(2)} />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 11, color: 'var(--muted)' }} formatter={(v: string) => FAULT_LABELS[v] ?? v} />
                {['normal', ...FAULT_ORDER].map((k) => <Scatter key={k} name={k} data={groups[k] ?? []} fill={FAULT_COLORS[k]} fillOpacity={k === 'normal' ? 0.28 : 0.75} isAnimationActive={false} />)}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <div className="space-y-4 xl:col-span-5">
          <Card>
            <CardHeader title="Cumulative explained variance" subtitle={Object.entries(d.report.components_for).map(([k, v]) => `${k}: ${v}`).join(' · ')} />
            <div className="px-2 pb-3">
              <ResponsiveContainer width="100%" height={170}>
                <LineChart data={cum} margin={{ top: 8, right: 14, left: -14, bottom: 0 }}>
                  <CartesianGrid vertical={false} /><XAxis dataKey="k" {...axis} /><YAxis domain={[0.3, 1]} tickFormatter={(v) => pct(v, 0)} {...axis} />
                  <Tooltip content={<ChartTooltip labelFormatter={(l: number) => `${l} components`} valueFormatter={(v: number) => pct(v)} />} />
                  <Line name="Variance" dataKey="v" stroke={ACC} strokeWidth={2.2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
          <Card>
            <CardHeader title="PCA vs random projection" subtitle="Same dimensionality, downstream fault classification" />
            <table className="w-full text-[12.5px]"><thead><tr className="eyebrow border-b border-line"><th className="px-4 py-2 text-left">Method</th><th className="text-right">Dims</th><th className="text-right">Macro-F1</th><th className="px-4 text-right">Recon. error</th></tr></thead>
              <tbody>{d.comparison.map((r: any) => <tr key={r.method} className="border-b border-line/60 last:border-0"><td className="px-4 py-2 font-medium">{r.method}</td><td className="text-right tnum text-fg-2">{r.dims}</td><td className="text-right tnum">{f3(r.macro_f1)}</td><td className="px-4 text-right tnum text-fg-2">{r.recon_error ? r.recon_error.toFixed(3) : '—'}</td></tr>)}</tbody></table>
          </Card>
        </div>
      </div>
    </Section>
  )
}

/* ───────────────────────────── Anomaly ───────────────────────────── */
function Anomaly({ an }: { an: any[] }) {
  const bars = an.map((m) => ({ name: m.model.replace(' (baseline)', '').replace('Static thresholds', 'Static thresholds'), Precision: m.precision, Recall: m.recall, F1: m.f1, base: m.key === 'threshold' }))
  const curves = useMemo(() => {
    const grid: Record<string, any> = {}
    an.filter((m) => m.pr_curve.length).forEach((m) => m.pr_curve.forEach(([r, p]: number[]) => { const k = Math.round(r * 50) / 50; (grid[k] ??= { recall: k })[m.model] = p }))
    return Object.values(grid).sort((a: any, b: any) => a.recall - b.recall)
  }, [an])
  const palette = ['var(--accent)', 'var(--accent-2)', 'var(--warn)']
  return (
    <Section id="anomaly" eyebrow="Module 3 · Unsupervised anomaly detection" title="Learning “normal” beats fixed thresholds" desc="Detectors are trained on normal windows only — including healthy traffic spikes and backups — and calibrated to an alert budget of at most 0.5 sustained false alarms per hour on validation data. The baseline is a fixed set of Prometheus-style alert rules.">
      <div className="grid gap-4 xl:grid-cols-12">
        <Card className="xl:col-span-7">
          <CardHeader title="Window-level precision / recall / F1" subtitle="Test split, grace windows around fault onset and recovery excluded" />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={bars} margin={{ top: 8, right: 10, left: -14, bottom: 0 }} barGap={3}>
                <CartesianGrid vertical={false} /><XAxis dataKey="name" {...axis} interval={0} /><YAxis domain={[0, 1]} {...axis} />
                <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTooltip valueFormatter={(v: number) => v.toFixed(3)} />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Precision" fill="var(--info)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="Recall" fill="var(--accent-2)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="F1" fill="var(--accent)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card className="xl:col-span-5">
          <CardHeader title="Precision–recall curves" subtitle="Score-based, threshold-free" />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={curves as any} margin={{ top: 8, right: 14, left: -14, bottom: 0 }}>
                <CartesianGrid vertical={false} /><XAxis dataKey="recall" type="number" domain={[0, 1]} {...axis} /><YAxis domain={[0.3, 1]} {...axis} />
                <Tooltip content={<ChartTooltip labelFormatter={(l: number) => `recall ${l.toFixed(2)}`} valueFormatter={(v: number) => v.toFixed(3)} />} />
                <Legend iconType="line" wrapperStyle={{ fontSize: 11 }} />
                {an.filter((m) => m.pr_curve.length).map((m, i) => <Line key={m.model} dataKey={m.model} stroke={palette[i % 3]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <Card>
        <div className="overflow-x-auto px-2 py-2">
          <table className="w-full min-w-[820px] text-left text-[12.5px]">
            <thead><tr className="eyebrow border-b border-line"><th className="px-3 py-2">Detector</th><th>F1</th><th>AUC</th><th>Episodes detected</th><th>Mean delay</th><th>False alarms / h</th>{FAULT_ORDER.map((f) => <th key={f} className="text-center">{FAULT_LABELS[f].split(' ')[0]}</th>)}</tr></thead>
            <tbody>
              {an.map((m) => (
                <tr key={m.model} className="border-b border-line/60 last:border-0">
                  <td className="px-3 py-2.5 font-medium">{m.model}</td>
                  <td className="tnum font-semibold">{f3(m.f1)}</td><td className="tnum text-fg-2">{m.auc_roc ? f3(m.auc_roc) : '—'}</td>
                  <td className="tnum">{m.detected}/{m.episodes}</td><td className="tnum text-fg-2">{Math.round(m.mean_delay_s)} s</td>
                  <td className="tnum">{m.false_alarms_per_hour.toFixed(2)}</td>
                  {FAULT_ORDER.map((f) => <td key={f} className="text-center tnum" style={heat(m.per_fault_recall[f])}>{pct(m.per_fault_recall[f], 0)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="px-4 pb-3 text-[11.5px] text-muted">Per-fault columns = share of fault windows flagged. Static thresholds miss anything that does not cross a fixed limit — database slowdowns and cache failures in particular.</p>
      </Card>
    </Section>
  )
}

/* ───────────────────────────── Classification ───────────────────────────── */
function Classification({ cl }: { cl: any }) {
  const [sel, setSel] = useState<string>(cl.best_ensemble)
  const models = cl.models as any[]
  const m = models.find((x) => x.model === sel)!
  const bars = models.filter((x) => x.family !== 'baseline').map((x) => ({ name: x.model, f1: x.macro_f1, fam: x.family, t: x.train_seconds }))
  const classes: string[] = cl.classes
  const cm: number[][] = m.confusion
  return (
    <Section id="classification" eyebrow="Modules 4 & 5 · Classification and ensembles" title="Which fault is it? Single models vs ensembles" desc="Fault-type classification on anomalous windows. Cross-validation inside the stacking ensemble is grouped by experiment run, so a run never appears on both sides of a fold. The majority-class baseline scores near zero macro-F1.">
      <div className="grid gap-4 xl:grid-cols-12">
        <Card className="xl:col-span-6">
          <CardHeader title="Macro-F1 by model" subtitle="Test split · click a bar to inspect its confusion matrix" />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={330}>
              <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 40, left: 40, bottom: 0 }}>
                <CartesianGrid horizontal={false} /><XAxis type="number" domain={[0.7, 1]} {...axis} /><YAxis type="category" dataKey="name" width={118} {...axis} tick={{ fill: 'var(--fg-2, #aab)', fontSize: 11.5 }} />
                <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTooltip valueFormatter={(v: number) => v.toFixed(3)} />} />
                <Bar name="Macro-F1" dataKey="f1" radius={[0, 5, 5, 0]} isAnimationActive={false} onClick={(d: any) => setSel(d.name)} label={{ position: 'right', fill: 'var(--text-2)', fontSize: 11, formatter: (v: number) => v.toFixed(3) }}>
                  {bars.map((x) => <Cell key={x.name} cursor="pointer" fill={x.fam === 'ensemble' ? 'var(--accent)' : 'var(--accent-2)'} fillOpacity={x.name === sel ? 1 : 0.6} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-4 px-3 pb-1 text-[11px] text-muted"><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-accent-2" />Single</span><span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-accent" />Ensemble</span></div>
          </div>
        </Card>
        <Card className="xl:col-span-6">
          <CardHeader title={`Confusion matrix · ${sel}`} subtitle={`Rows = true fault, columns = predicted · accuracy ${pct(m.accuracy)} · trained in ${m.train_seconds}s`} actions={<select value={sel} onChange={(e) => setSel(e.target.value)} className="h-7 rounded-lg border border-line-strong bg-surface-2 px-2 text-[12px] outline-none">{models.map((x) => <option key={x.model}>{x.model}</option>)}</select>} />
          <div className="overflow-x-auto px-4 pb-4">
            <table className="w-full min-w-[460px] border-separate border-spacing-1 text-center text-[11.5px]">
              <thead><tr><th />{classes.map((c) => <th key={c} className="px-1 pb-1 text-[10.5px] font-medium text-muted">{FAULT_LABELS[c].split(' ')[0]}</th>)}</tr></thead>
              <tbody>
                {cm.map((row, i) => {
                  const tot = row.reduce((a, b) => a + b, 0) || 1
                  return (
                    <tr key={i}>
                      <th className="whitespace-nowrap pr-2 text-right text-[11px] font-medium text-fg-2">{FAULT_LABELS[classes[i]]}</th>
                      {row.map((v, j) => <td key={j} className="rounded-md py-2 font-medium tnum" style={{ ...heat(v / tot, i === j ? 'accent' : 'ok'), ...(i !== j && v / tot > 0.02 ? { background: `color-mix(in oklab, var(--crit) ${Math.round(Math.min(v / tot, 1) * 80)}%, transparent)` } : {}), color: v / tot > 0.35 ? '#fff' : 'var(--text-2)' }}>{v ? (v / tot > 0.005 ? Math.round((v / tot) * 100) : '<1') : '·'}</td>)}
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <p className="mt-2 text-[11px] text-muted">Cells show % of each true class. Per-class F1: {classes.map((c) => `${FAULT_LABELS[c].split(' ')[0]} ${m.per_class_f1[c].toFixed(2)}`).join(' · ')}</p>
          </div>
        </Card>
      </div>
      <Card>
        <div className="overflow-x-auto px-2 py-2">
          <table className="w-full min-w-[640px] text-left text-[12.5px]">
            <thead><tr className="eyebrow border-b border-line"><th className="px-3 py-2">Model</th><th>Family</th><th>Macro-F1 (test)</th><th>Macro-F1 (val)</th><th>Accuracy</th><th>Train time</th><th>Inference / window</th></tr></thead>
            <tbody>{models.map((x) => <tr key={x.model} onClick={() => setSel(x.model)} className="cursor-pointer border-b border-line/60 last:border-0 hover:bg-surface-2"><td className="px-3 py-2 font-medium">{x.model}</td><td className="capitalize text-fg-2">{x.family}</td><td className="tnum font-semibold">{f3(x.macro_f1)}</td><td className="tnum text-fg-2">{f3(x.val_macro_f1)}</td><td className="tnum text-fg-2">{pct(x.accuracy)}</td><td className="tnum text-fg-2">{x.train_seconds}s</td><td className="tnum text-fg-2">{x.infer_ms_per_window.toFixed(3)} ms</td></tr>)}</tbody>
          </table>
        </div>
      </Card>
    </Section>
  )
}

/* ───────────────────────────── Clustering ───────────────────────────── */
function Clustering({ c }: { c: any }) {
  const [by, setBy] = useState<'cluster' | 'truth'>('cluster')
  const series = useMemo(() => {
    const g: Record<string, any[]> = {}
    c.map.forEach((p: any) => (g[by === 'cluster' ? `c${p.cluster}` : p.label] ??= []).push(p))
    return g
  }, [c, by])
  return (
    <Section id="clustering" eyebrow="Module 6 · Clustering" title="Recurring failure patterns, found without labels" desc="K-means, Ward hierarchical clustering and DBSCAN group the windows flagged by the detector. Fault labels are used only afterwards, to check whether the discovered groups correspond to real failure modes (adjusted Rand index, NMI).">
      <div className="grid gap-4 xl:grid-cols-12">
        <Card className="xl:col-span-7">
          <CardHeader title="Test windows in PCA space" subtitle="Assigned by the deployed K-means model" actions={<Segmented size="sm" value={by} onChange={setBy} options={[{ value: 'cluster', label: 'Cluster' }, { value: 'truth', label: 'True fault' }]} />} />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart margin={{ top: 8, right: 16, left: -10, bottom: 0 }}>
                <CartesianGrid /><XAxis type="number" dataKey="x" domain={clipDomain(c.map, 'x')} allowDataOverflow {...axis} /><YAxis type="number" dataKey="y" domain={clipDomain(c.map, 'y')} allowDataOverflow {...axis} /><ZAxis range={[14, 14]} />
                {Object.entries(series).map(([k, d]) => {
                  const col = by === 'cluster' ? CLUSTER_COLORS[Number(k.slice(1)) % CLUSTER_COLORS.length] : FAULT_COLORS[k]
                  return <Scatter key={k} data={d} fill={col} fillOpacity={k === 'normal' ? 0.25 : 0.75} isAnimationActive={false} />
                })}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card className="xl:col-span-5">
          <CardHeader title="Algorithms vs random assignment" subtitle={`Fit on ${c.n_flagged_test.toLocaleString()} flagged test windows`} />
          <table className="w-full text-[12.5px]"><thead><tr className="eyebrow border-b border-line"><th className="px-4 py-2 text-left">Algorithm</th><th className="text-right">k</th><th className="text-right">Silhouette</th><th className="text-right">ARI</th><th className="px-4 text-right">NMI</th></tr></thead>
            <tbody>{c.algorithms.map((a: any) => <tr key={a.model} className="border-b border-line/60 last:border-0"><td className="px-4 py-2.5 font-medium">{a.model}</td><td className="text-right tnum text-fg-2">{a.n_clusters}</td><td className="text-right tnum">{f3(a.silhouette)}</td><td className="text-right font-semibold tnum" style={a.model.startsWith('Random') ? { color: 'var(--muted)' } : { color: 'var(--accent)' }}>{f3(a.ari)}</td><td className="px-4 text-right tnum text-fg-2">{f3(a.nmi)}</td></tr>)}</tbody></table>
          <div className="px-4 py-3 text-[11.5px] text-muted">k chosen by silhouette over 4–12: {Object.entries(c.k_scores).map(([k, v]: any) => `${k}→${v.toFixed(2)}`).join('  ')}</div>
        </Card>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {c.clusters.slice(0, 8).map((k: any) => (
          <div key={k.cluster} className="card p-3.5">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 text-[13px] font-semibold"><span className="h-2.5 w-2.5 rounded-full" style={{ background: CLUSTER_COLORS[k.cluster % CLUSTER_COLORS.length] }} />Pattern {k.cluster + 1}</span>
              <span className="text-[11px] text-muted tnum">{k.size} windows</span>
            </div>
            <div className="mt-1.5 text-[12px] text-fg-2">Matches <b className="text-fg">{FAULT_LABELS[k.dominant] ?? k.dominant}</b> · purity {pct(k.purity, 0)}</div>
            <ul className="mt-2 space-y-0.5 text-[11.5px] text-muted">{k.signature.map((s: any) => <li key={s.feature} className="flex justify-between"><span className="truncate">{s.label}</span><span className="tnum" style={{ color: s.z > 0 ? 'var(--warn)' : 'var(--info)' }}>{s.z > 0 ? '+' : ''}{s.z}σ</span></li>)}</ul>
          </div>
        ))}
      </div>
    </Section>
  )
}

/* ───────────────────────────── Early warning ───────────────────────────── */
function Warning({ ew, hist, histG }: { ew: any[]; hist: any[]; histG: any[] }) {
  const lstm = ew.find((m) => m.model === 'LSTM')
  const base = ew.find((m) => m.model.startsWith('Static'))
  const bins = useMemo(() => {
    const edges = [0, 30, 60, 90, 120, 180, 240, 600]
    const label = (i: number) => `${edges[i]}–${edges[i + 1] === 600 ? '+' : edges[i + 1]}s`
    const rows: any[] = [{ bin: 'Missed' }, { bin: 'Late' }, ...edges.slice(0, -1).map((_, i) => ({ bin: label(i) }))]
    const fill = (m: any, key: string) => {
      rows[0][key] = m.missed; rows[1][key] = m.late
      for (let i = 0; i < edges.length - 1; i++) rows[i + 2][key] = m.leads.filter((l: number) => l >= edges[i] && l < edges[i + 1]).length
    }
    fill(lstm, 'OpsMind LSTM'); fill(base, 'Static thresholds')
    return rows
  }, [lstm, base])
  const epochs = hist.map((h, i) => ({ epoch: h.epoch, LSTM: h.val_ap, GRU: histG[i]?.val_ap }))
  return (
    <Section id="warning" eyebrow="Module 7 · Early warning" title="Predicting incidents before users feel them" desc="An LSTM reads the last five minutes of compressed features and outputs the probability that user impact starts within two minutes. It is compared with a GRU, a gradient-boosting fallback, an Isolation-Forest alert and the static-threshold alert time. Probabilities are Platt-calibrated on validation data; the alert threshold respects a false-alarm budget.">
      <div className="grid gap-4 xl:grid-cols-12">
        <Card className="xl:col-span-7">
          <CardHeader title="Lead time distribution" subtitle="Seconds between the first alert and the start of user impact (test episodes with impact)" />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={290}>
              <BarChart data={bins} margin={{ top: 8, right: 10, left: -18, bottom: 0 }} barGap={2}>
                <CartesianGrid vertical={false} /><XAxis dataKey="bin" {...axis} interval={0} /><YAxis allowDecimals={false} {...axis} />
                <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTooltip valueFormatter={(v: number) => `${v} episodes`} />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="OpsMind LSTM" fill="var(--accent)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="Static thresholds" fill={GREY} radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card className="xl:col-span-5">
          <CardHeader title="Training curve" subtitle="Validation average precision per epoch" />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={290}>
              <LineChart data={epochs} margin={{ top: 8, right: 14, left: -12, bottom: 0 }}>
                <CartesianGrid vertical={false} /><XAxis dataKey="epoch" {...axis} /><YAxis domain={['auto', 'auto']} tickFormatter={(v) => Number(v).toFixed(3)} {...axis} />
                <Tooltip content={<ChartTooltip labelFormatter={(l: number) => `epoch ${l}`} valueFormatter={(v: number) => v.toFixed(4)} />} />
                <Legend iconType="line" wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="LSTM" stroke="var(--accent)" strokeWidth={2.2} dot={false} isAnimationActive={false} />
                <Line dataKey="GRU" stroke="var(--accent-2)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
      <Card>
        <div className="overflow-x-auto px-2 py-2">
          <table className="w-full min-w-[900px] text-left text-[12.5px]">
            <thead><tr className="eyebrow border-b border-line"><th className="px-3 py-2">Alerting method</th><th>Warned</th><th>Late</th><th>Missed</th><th>Recall</th><th>Precision</th><th>Mean lead</th><th>False alarms / h</th><th>Window AP</th>{FAULT_ORDER.map((f) => <th key={f} className="text-center">{FAULT_LABELS[f].split(' ')[0]}</th>)}</tr></thead>
            <tbody>{ew.map((m) => (
              <tr key={m.model} className="border-b border-line/60 last:border-0">
                <td className="px-3 py-2.5 font-medium">{m.model}{m.model === 'LSTM' && <span className="ml-2 rounded bg-accent-soft px-1.5 py-[1px] text-[10px] font-semibold text-accent">DEPLOYED</span>}</td>
                <td className="tnum">{m.warned}/{m.eligible_episodes}</td><td className="tnum text-fg-2">{m.late}</td><td className="tnum text-fg-2">{m.missed}</td>
                <td className="font-semibold tnum">{pct(m.recall, 0)}</td><td className="tnum">{pct(m.precision, 0)}</td><td className="tnum">{Math.round(m.mean_lead_s)} s</td>
                <td className="tnum">{m.false_alarms_per_hour.toFixed(2)}</td><td className="tnum text-fg-2">{m.window_ap ? m.window_ap.toFixed(3) : '—'}</td>
                {FAULT_ORDER.map((f) => <td key={f} className="text-center tnum" style={heat(m.per_fault[f]?.recall)}>{m.per_fault[f] ? pct(m.per_fault[f].recall, 0) : '—'}</td>)}
              </tr>))}</tbody>
          </table>
        </div>
        <p className="px-4 pb-3 text-[11.5px] text-muted">Sudden faults (a container kill with no crash-loop, a Redis stop) have no precursor, so no model can warn about them — only gradual degradation (database, memory, network, CPU) can be predicted. Reported per fault type instead of hidden in an average.</p>
      </Card>
    </Section>
  )
}

/* ───────────────────────────── Root cause ───────────────────────────── */
function RootCause({ methods }: { methods: any[] }) {
  const data = methods.map((m) => ({ name: m.method.replace(' (baseline)', ''), 'Top-1': m.episode.top1, 'Top-3': m.episode.top3 }))
  const comps = Object.keys(methods[0].by_fault)
  return (
    <Section id="rca" eyebrow="Module 8 · Root-cause ranking" title="Pointing at the faulty component" desc="SHAP attributions of the classifier are summed per component, then the dependency graph is walked: anomalous components that others depend on are boosted, and components whose own dependencies are anomalous are discounted. Output is a ranked list of likely causes — not a proven causal chain.">
      <div className="grid gap-4 xl:grid-cols-12">
        <Card className="xl:col-span-6">
          <CardHeader title="Episode-level accuracy" subtitle={`Test fault episodes (n = ${methods[0].episode.n}) · ablation from baseline to full method`} />
          <div className="px-2 pb-3">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data} margin={{ top: 8, right: 10, left: -14, bottom: 0 }} barGap={3}>
                <CartesianGrid vertical={false} /><XAxis dataKey="name" {...axis} interval={0} tick={{ fill: 'var(--muted)', fontSize: 10 }} /><YAxis domain={[0, 1]} {...axis} />
                <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ChartTooltip valueFormatter={(v: number) => pct(v)} />} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Top-1" fill="var(--accent)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="Top-3" fill="var(--accent-2)" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
        <Card className="xl:col-span-6">
          <CardHeader title="Top-1 accuracy by fault type" subtitle="Where each method gets it right" />
          <div className="overflow-x-auto px-2 pb-3">
            <table className="w-full min-w-[520px] text-[12.5px]">
              <thead><tr className="eyebrow border-b border-line"><th className="px-3 py-2 text-left">Method</th>{comps.map((f) => <th key={f} className="text-center">{FAULT_LABELS[f].split(' ')[0]}</th>)}</tr></thead>
              <tbody>{methods.map((m) => <tr key={m.method} className="border-b border-line/60 last:border-0"><td className="px-3 py-2.5 font-medium">{m.method.replace(' (baseline)', '')}</td>{comps.map((f) => <td key={f} className="text-center tnum" style={heat(m.by_fault[f]?.top1)}>{pct(m.by_fault[f]?.top1, 0)}</td>)}</tr>)}</tbody>
            </table>
            <p className="px-3 pt-3 text-[11.5px] text-muted">Targets include databases and caches, so “highest raw CPU” — what an on-call engineer often checks first — misses most incidents. Components: {Object.values(COMPONENT_LABELS).join(', ')}.</p>
          </div>
        </Card>
      </div>
    </Section>
  )
}

/* ───────────────────────────── Real testbed check ───────────────────────────── */
function RealCheck() {
  const [r, setR] = useState<any>(undefined)
  const nav = useNavigate()
  useEffect(() => { get('/api/real').then(setR).catch(() => setR(null)) }, [])
  if (!r) return null
  const d = r.detection, ew = r.early_warning, ds = r.dataset, c = r.classification
  const Row = ({ label, a, b, c: v3, fmt = (v: any) => String(v) }: any) => (
    <tr className="border-b border-line/60 last:border-0"><td className="px-4 py-2.5 text-fg-2">{label}</td><td className="text-right tnum">{fmt(a)}</td><td className="text-right tnum">{fmt(b)}</td><td className="px-4 text-right font-semibold tnum" style={{ color: 'var(--accent)' }}>{v3 === undefined ? '—' : fmt(v3)}</td></tr>
  )
  const n = (x: any) => `${x.warned}/${x.eligible_episodes}`
  return (
    <Section id="real" eyebrow="Real Docker testbed" title="Simulation vs reality, measured" desc={<>{ds.runs} real runs from two overnight campaigns on the Docker Compose stack ({ds.fault_episodes} fault episodes of 6 types plus {ds.normal_runs} healthy controls, {ds.windows.toLocaleString()} windows), labelled by the real fault injector. Whole runs are split chronologically; everything below is scored on the same <b>{r.test_runs} held-out real runs</b> ({r.test_fault_episodes} fault episodes). “Simulator” = models trained only on simulated data. “Trained on real” = the same pipeline trained on the {ds.runs_by_split.train} real training runs. Small sample: expect noise. {ds.overload_windows} windows where the user SLO was breached with <b>no fault injected</b> (an over-heavy 4× burst overloading the single-CPU gateway, and one run that started on a degraded gateway) are excluded from training and scoring.</>}>
      <div className="grid gap-4 xl:grid-cols-12">
        <Card className="xl:col-span-7">
          <CardHeader title="Static thresholds vs simulator-trained vs trained on real" subtitle="Held-out real runs" />
          <table className="w-full text-[12.5px]"><thead><tr className="eyebrow border-b border-line"><th className="px-4 py-2 text-left">Metric</th><th className="text-right">Static</th><th className="text-right">Simulator</th><th className="px-4 text-right">Trained on real</th></tr></thead><tbody>
            <Row label="Detection F1" a={d.static.f1} b={d.simulator.f1} c={d.real.f1} fmt={f3} />
            <Row label="Precision" a={d.static.precision} b={d.simulator.precision} c={d.real.precision} fmt={f3} />
            <Row label="Recall" a={d.static.recall} b={d.simulator.recall} c={d.real.recall} fmt={f3} />
            <Row label="Fault episodes detected" a={`${d.static.detected}/${r.test_fault_episodes}`} b={`${d.simulator.detected}/${r.test_fault_episodes}`} c={`${d.real.detected}/${r.test_fault_episodes}`} />
            <Row label="False alarms / hour" a={d.static.false_alarms_per_hour} b={d.simulator.false_alarms_per_hour} c={d.real.false_alarms_per_hour} fmt={(v: number) => v.toFixed(1)} />
            <Row label="Warned before user impact" a={n(ew.static)} b={n(ew.simulator)} c={n(ew.real)} />
            <Row label="Fault-type macro-F1" a="—" b="—" c={f3(c.best_single.f1)} />
            <Row label="Root cause top-1 / top-3" a={`${pct(r.rootcause.baseline.top1, 0)} / ${pct(r.rootcause.baseline.top3, 0)} (raw CPU)`} b="—" c={`${pct(r.rootcause.real.top1, 0)} / ${pct(r.rootcause.real.top3, 0)}`} />
          </tbody></table>
          <p className="px-4 py-3 text-[11.5px] leading-relaxed text-muted">Trained-on-real detector: {d.real.model}. Detector + classifier on real: F1 {f3(c.pipeline.detector_plus_classifier.f1)}, false alarms {c.pipeline.detector_plus_classifier.false_alarms_per_hour.toFixed(1)}/h (from {c.pipeline.detector_only.false_alarms_per_hour.toFixed(1)}/h). Early warning works on real data for <b>gradual</b> faults only: the LSTM warned for the database cascades with a long lead time (see the warning row), but fast faults (network degradation, crash-loops) reach users within seconds and cannot be predicted. Only {ew.real.eligible_episodes} test episodes reached users. K-means clustering still recovers real failure patterns (ARI {f3(r.clustering.ari)}).</p>
        </Card>
        <Card className="xl:col-span-5">
          <CardHeader title="Held-out real episodes" subtitle="Replay them with the real-trained models" />
          <div className="overflow-x-auto px-2 pb-3"><table className="w-full min-w-[460px] text-left text-[12.5px]">
            <thead><tr className="eyebrow border-b border-line"><th className="px-3 py-2">Injected</th><th>Target</th><th>Classified</th><th>Root cause</th><th /></tr></thead>
            <tbody>{r.episodes.map((e: any) => (
              <tr key={e.run_id} className="border-b border-line/60 last:border-0 hover:bg-surface-2">
                <td className="px-3 py-2"><span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: FAULT_COLORS[e.fault_type] }} />{FAULT_LABELS[e.fault_type]}</span></td>
                <td>{COMPONENT_LABELS[e.target as keyof typeof COMPONENT_LABELS]}</td>
                <td><Badge tone={e.predicted_type === e.fault_type ? 'ok' : 'crit'}>{FAULT_LABELS[e.predicted_type]}</Badge></td>
                <td>{e.rca_top3.slice(0, 2).map((x: string, k: number) => <span key={x} className={`mr-1.5 ${x === e.target ? 'font-semibold text-ok' : k === 0 ? 'font-semibold' : 'text-muted'}`}>{COMPONENT_LABELS[x as keyof typeof COMPONENT_LABELS]}</span>)}</td>
                <td className="pr-3 text-right"><Button size="sm" variant="ghost" icon={<Film size={12} />} onClick={() => actions.replay(e.run_id).then(() => nav('/'))}>Replay</Button></td>
              </tr>))}</tbody>
          </table></div>
        </Card>
      </div>
    </Section>
  )
}
