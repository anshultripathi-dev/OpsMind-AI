import { Network } from 'lucide-react'
import { Card, CardHeader } from '@/components/ui'
import { useLive } from '@/lib/store'
import { COMPONENT_COLORS, COMPONENT_LABELS, TONE_VAR, healthTone } from '@/lib/colors'
import type { ComponentId } from '@/lib/types'

const POS: Record<ComponentId, [number, number]> = {
  gateway: [92, 200],
  auth: [305, 88],
  order: [305, 318],
  redis: [530, 34],
  payment: [530, 368],
  postgres: [722, 200],
}
const W = 150
const H = 56
const KIND: Record<ComponentId, string> = {
  gateway: 'FastAPI · ingress',
  auth: 'FastAPI service',
  order: 'FastAPI service',
  payment: 'FastAPI service',
  postgres: 'Database',
  redis: 'Cache',
}

function edgePath(a: ComponentId, b: ComponentId) {
  const [ax, ay] = POS[a]
  const [bx, by] = POS[b]
  const x1 = ax + W / 2
  const x2 = bx - W / 2
  const mx = (x1 + x2) / 2
  return `M ${x1} ${ay} C ${mx} ${ay}, ${mx} ${by}, ${x2} ${by}`
}

export function Topology() {
  const { ticks, config, selected, setSelected, showTruth } = useLive()
  const last = ticks[ticks.length - 1]
  const rank: Partial<Record<ComponentId, number>> = {}
  const rcaActive = !!last?.rca && (last.anomaly.flagged || last.risk.alert || last.risk.prob > 0.3)
  if (rcaActive) last!.rca!.slice(0, 3).forEach((r, i) => (rank[r.component] = i + 1))
  const truthTargets = new Set(showTruth ? last?.truth?.faults.map((f) => f.target) : [])

  return (
    <Card className="flex h-full flex-col animate-fade-up">
      <CardHeader
        icon={<Network size={15} />}
        title="Service topology"
        subtitle="Colour = deviation from normal behaviour · numbered badges = root-cause ranking"
        actions={
          <div className="hidden items-center gap-3 text-[11px] text-muted sm:flex">
            {(['ok', 'warn', 'crit'] as const).map((t) => (
              <span key={t} className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ background: TONE_VAR[t] }} />
                {t === 'ok' ? 'Normal' : t === 'warn' ? 'Degraded' : 'Anomalous'}
              </span>
            ))}
          </div>
        }
      />
      <div className="grid-bg relative flex-1 px-2 pb-3">
        <svg viewBox="0 0 800 404" className="h-full w-full" role="img" aria-label="Service dependency graph">
          <defs>
            <marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0 0 L8 4 L0 8 z" fill="var(--border-strong)" />
            </marker>
            <marker id="arr-w" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0 0 L8 4 L0 8 z" fill="var(--warn)" />
            </marker>
            <marker id="arr-c" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0 0 L8 4 L0 8 z" fill="var(--crit)" />
            </marker>
          </defs>

          {config?.edges.map((e) => {
            const dev = last ? last.deviation[e.target] : 0
            const tone = healthTone(dev)
            const rps = last?.sample[e.target]?.rps ?? 0
            const down = last && last.sample[e.target]?.up === 0
            const color = down ? 'var(--crit)' : tone === 'ok' ? 'var(--border-strong)' : TONE_VAR[tone]
            const marker = down || tone === 'crit' ? 'url(#arr-c)' : tone === 'warn' ? 'url(#arr-w)' : 'url(#arr)'
            const speed = Math.max(0.5, Math.min(2.4, 2.4 - rps / 220))
            return (
              <g key={`${e.source}-${e.target}`}>
                <path d={edgePath(e.source, e.target)} fill="none" stroke={color} strokeWidth={1.4 + Math.min(e.fan, 1.2)} opacity={0.35} markerEnd={marker} />
                {!down && <path d={edgePath(e.source, e.target)} fill="none" stroke={color} strokeWidth={1.6} strokeDasharray="5 7" style={{ animation: `flow ${speed}s linear infinite` }} opacity={tone === 'ok' ? 0.7 : 1} />}
              </g>
            )
          })}

          {(Object.keys(POS) as ComponentId[]).map((id) => {
            const [cx, cy] = POS[id]
            const s = last?.sample[id]
            const dev = last?.deviation[id] ?? 0
            const down = s?.up === 0
            const tone = down ? 'crit' : healthTone(dev)
            const col = TONE_VAR[tone]
            const r = rank[id]
            const sel = selected === id
            return (
              <g key={id} transform={`translate(${cx - W / 2}, ${cy - H / 2})`} onClick={() => setSelected(id)} style={{ cursor: 'pointer' }}>
                {r === 1 && <rect x={-4} y={-4} width={W + 8} height={H + 8} rx={16} fill="none" stroke="var(--crit)" strokeWidth={1.5} opacity={0.5} style={{ animation: 'blink 1.4s ease-in-out infinite' }} />}
                <rect width={W} height={H} rx={12} fill="var(--surface-2)" stroke={sel ? 'var(--accent)' : tone === 'ok' ? 'var(--border-strong)' : col} strokeWidth={sel ? 2 : 1.3} style={{ filter: tone !== 'ok' ? `drop-shadow(0 0 10px color-mix(in oklab, ${col} 35%, transparent))` : undefined, transition: 'stroke 0.3s' }} />
                <rect x={0} y={10} width={3.5} height={H - 20} rx={2} fill={COMPONENT_COLORS[id]} />
                <text x={16} y={20} fontSize={12.5} fontWeight={600} fill="var(--text)">{COMPONENT_LABELS[id]}</text>
                <text x={16} y={34} fontSize={9.5} fill="var(--muted)">{KIND[id]}</text>
                <text x={16} y={48} fontSize={10} fill="var(--text-2)" fontFamily="var(--font-mono)">
                  {down ? 'DOWN' : s ? `cpu ${s.cpu.toFixed(0)}% · ${s.lat_p95 < 1000 ? s.lat_p95.toFixed(0) : (s.lat_p95 / 1000).toFixed(1) + 'k'}ms` : '—'}
                </text>
                <circle cx={W - 14} cy={16} r={5} fill={col} style={{ filter: `drop-shadow(0 0 5px ${col})`, transition: 'fill 0.3s' }} />
                {truthTargets.has(id) && (
                  <g transform={`translate(${W - 44}, ${H - 18})`}>
                    <rect width={38} height={14} rx={7} fill="var(--crit-soft)" stroke="var(--crit)" strokeWidth={0.8} />
                    <text x={19} y={10.2} fontSize={8.5} fontWeight={700} textAnchor="middle" fill="var(--crit)">INJECTED</text>
                  </g>
                )}
                {r && (
                  <g transform={`translate(${W - 8}, -8)`}>
                    <circle r={11} fill={r === 1 ? 'var(--crit)' : 'var(--surface-3)'} stroke={r === 1 ? 'var(--crit)' : 'var(--border-strong)'} />
                    <text y={4} textAnchor="middle" fontSize={11} fontWeight={700} fill={r === 1 ? '#fff' : 'var(--text-2)'}>{r}</text>
                  </g>
                )}
              </g>
            )
          })}
        </svg>
      </div>
    </Card>
  )
}
