import type { ReactNode } from 'react'
import { Badge } from '@/components/ui'

export function Section({ id, eyebrow, title, desc, children }: { id: string; eyebrow: string; title: string; desc: ReactNode; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 space-y-3">
      <div className="max-w-3xl">
        <div className="eyebrow text-accent">{eyebrow}</div>
        <h2 className="mt-0.5 text-[18px] font-semibold tracking-tight">{title}</h2>
        <p className="mt-1 text-[13px] leading-relaxed text-fg-2">{desc}</p>
      </div>
      {children}
    </section>
  )
}

export function Criterion({ ok, title, value, detail }: { ok: boolean | null; title: string; value: string; detail: string }) {
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between">
        <span className="eyebrow">{title}</span>
        {ok === null ? <Badge>info</Badge> : <Badge tone={ok ? 'ok' : 'warn'} dot>{ok ? 'Met' : 'Not met'}</Badge>}
      </div>
      <div className="mt-2 text-[24px] font-semibold leading-none tracking-tight tnum">{value}</div>
      <div className="mt-1.5 text-[11.5px] leading-snug text-muted">{detail}</div>
    </div>
  )
}

/** Heat cell: value in [0,1] mapped to accent alpha. */
export function heat(v: number | null | undefined, hue: 'accent' | 'ok' = 'accent'): React.CSSProperties {
  if (v === null || v === undefined) return { color: 'var(--muted)' }
  const c = hue === 'ok' ? 'var(--ok)' : 'var(--accent)'
  return { background: `color-mix(in oklab, ${c} ${Math.round(Math.min(Math.max(v, 0), 1) * 62)}%, transparent)` }
}

export const pct = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`)
export const f3 = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(3))
