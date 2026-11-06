import clsx from 'clsx'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { TONE_VAR } from '@/lib/colors'

/* ── Card ─────────────────────────────────────────────────────────────────── */
export function Card({ className, children, ...rest }: { className?: string; children: ReactNode } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={clsx('card', className)} {...rest}>
      {children}
    </div>
  )
}

export function CardHeader({
  title,
  subtitle,
  icon,
  actions,
  className,
}: {
  title: ReactNode
  subtitle?: ReactNode
  icon?: ReactNode
  actions?: ReactNode
  className?: string
}) {
  return (
    <div className={clsx('flex items-start justify-between gap-3 px-4 pt-3.5 pb-2', className)}>
      <div className="flex min-w-0 items-center gap-2.5">
        {icon && <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-surface-3 text-fg-2">{icon}</span>}
        <div className="min-w-0">
          <h3 className="truncate text-[13px] font-semibold tracking-tight text-fg">{title}</h3>
          {subtitle && <p className="truncate text-[11.5px] text-muted">{subtitle}</p>}
        </div>
      </div>
      {actions && <div className="flex shrink-0 items-center gap-1.5">{actions}</div>}
    </div>
  )
}

/* ── Badge ────────────────────────────────────────────────────────────────── */
type Tone = 'ok' | 'warn' | 'crit' | 'info' | 'neutral' | 'accent'
const BADGE: Record<Tone, string> = {
  ok: 'bg-ok-soft text-ok',
  warn: 'bg-warn-soft text-warn',
  crit: 'bg-crit-soft text-crit',
  info: 'bg-info-soft text-info',
  accent: 'bg-accent-soft text-accent',
  neutral: 'bg-surface-3 text-fg-2',
}
export function Badge({ tone = 'neutral', children, dot, className }: { tone?: Tone; children: ReactNode; dot?: boolean; className?: string }) {
  return (
    <span className={clsx('inline-flex items-center gap-1.5 rounded-full px-2 py-[2px] text-[11px] font-medium leading-4', BADGE[tone], className)}>
      {dot && <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      {children}
    </span>
  )
}

/* ── Button ───────────────────────────────────────────────────────────────── */
type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'ghost' | 'outline' | 'danger'
  size?: 'sm' | 'md'
  icon?: ReactNode
}
export function Button({ variant = 'outline', size = 'md', icon, className, children, ...rest }: BtnProps) {
  return (
    <button
      className={clsx(
        'inline-flex select-none items-center justify-center gap-1.5 rounded-lg font-medium transition-all duration-150 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40',
        size === 'sm' ? 'h-7 px-2.5 text-[12px]' : 'h-8 px-3 text-[12.5px]',
        variant === 'primary' && 'bg-accent text-white shadow-[0_6px_20px_-8px_var(--accent)] hover:brightness-110',
        variant === 'outline' && 'border border-line-strong bg-surface-2 text-fg hover:border-accent/60 hover:bg-surface-3',
        variant === 'ghost' && 'text-fg-2 hover:bg-surface-3 hover:text-fg',
        variant === 'danger' && 'border border-crit/30 bg-crit-soft text-crit hover:bg-crit/20',
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </button>
  )
}

/* ── Segmented control ────────────────────────────────────────────────────── */
export function Segmented<T extends string | number>({
  value,
  onChange,
  options,
  size = 'md',
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: ReactNode }[]
  size?: 'sm' | 'md'
}) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={String(o.value)}
          onClick={() => onChange(o.value)}
          className={clsx(
            'rounded-md font-medium transition-colors',
            size === 'sm' ? 'px-2 py-[3px] text-[11.5px]' : 'px-2.5 py-1 text-[12px]',
            o.value === value ? 'bg-surface-3 text-fg shadow-sm' : 'text-muted hover:text-fg-2',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/* ── Sparkline ────────────────────────────────────────────────────────────── */
export function Spark({
  data,
  color = 'var(--accent)',
  height = 36,
  min,
  max,
  fill = true,
  threshold,
}: {
  data: number[]
  color?: string
  height?: number
  min?: number
  max?: number
  fill?: boolean
  threshold?: number
}) {
  const W = 200
  if (data.length < 2) return <div style={{ height }} />
  const lo = min ?? Math.min(...data)
  const hi = max ?? Math.max(...data)
  const span = hi - lo || 1
  const y = (v: number) => height - 2 - ((Math.min(Math.max(v, lo), hi) - lo) / span) * (height - 4)
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * W},${y(v)}`)
  const id = `sp${Math.abs(Math.round(data[0] * 997 + data.length))}${color.replace(/\W/g, '')}`
  return (
    <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" width="100%" height={height} className="overflow-visible">
      <defs>
        <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.28" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {threshold !== undefined && (
        <line x1="0" x2={W} y1={y(threshold)} y2={y(threshold)} stroke="var(--muted)" strokeDasharray="3 3" strokeWidth="1" opacity="0.6" vectorEffect="non-scaling-stroke" />
      )}
      {fill && <polygon points={`0,${height} ${pts.join(' ')} ${W},${height}`} fill={`url(#${id})`} />}
      <polyline points={pts.join(' ')} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

/* ── Radial gauge ─────────────────────────────────────────────────────────── */
export function Gauge({
  value,
  threshold,
  tone,
  size = 112,
  children,
}: {
  value: number
  threshold?: number
  tone: 'ok' | 'warn' | 'crit' | 'info'
  size?: number
  children?: ReactNode
}) {
  const r = 44
  const c = Math.PI * r // half circle
  const v = Math.min(Math.max(value, 0), 1)
  const ang = (f: number) => Math.PI * (1 - f)
  const tx = (f: number, rad: number) => 56 + rad * Math.cos(ang(f))
  const ty = (f: number, rad: number) => 56 - rad * Math.sin(ang(f))
  return (
    <div className="relative" style={{ width: size, height: size * 0.62 }}>
      <svg viewBox="0 0 112 70" width={size} height={size * 0.62}>
        <path d={`M ${56 - r} 56 A ${r} ${r} 0 0 1 ${56 + r} 56`} fill="none" stroke="var(--surface-3)" strokeWidth="9" strokeLinecap="round" />
        <path
          d={`M ${56 - r} 56 A ${r} ${r} 0 0 1 ${56 + r} 56`}
          fill="none"
          stroke={TONE_VAR[tone]}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={`${c * v} ${c}`}
          style={{ transition: 'stroke-dasharray 0.6s cubic-bezier(.2,.8,.2,1), stroke 0.3s', filter: `drop-shadow(0 0 6px ${TONE_VAR[tone]})` }}
        />
        {threshold !== undefined && (
          <line x1={tx(threshold, r - 8)} y1={ty(threshold, r - 8)} x2={tx(threshold, r + 8)} y2={ty(threshold, r + 8)} stroke="var(--fg-2, #aaa)" strokeWidth="1.6" style={{ stroke: 'var(--text-2)' }} />
        )}
      </svg>
      <div className="absolute inset-x-0 bottom-0 flex flex-col items-center leading-none">{children}</div>
    </div>
  )
}

/* ── Misc ─────────────────────────────────────────────────────────────────── */
export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx('skeleton', className)} />
}

export function Empty({ icon, title, hint, className }: { icon?: ReactNode; title: string; hint?: string; className?: string }) {
  return (
    <div className={clsx('flex flex-col items-center justify-center gap-2 px-6 py-8 text-center', className)}>
      {icon && <div className="grid h-10 w-10 place-items-center rounded-xl bg-surface-3 text-muted">{icon}</div>}
      <div className="text-[13px] font-medium text-fg-2">{title}</div>
      {hint && <div className="max-w-[28ch] text-[12px] text-muted">{hint}</div>}
    </div>
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line-strong bg-surface-2 px-1.5 py-[1px] font-mono text-[10.5px] text-muted">{children}</kbd>
}

export function Metric({ label, value, unit, tone }: { label: string; value: ReactNode; unit?: string; tone?: 'ok' | 'warn' | 'crit' }) {
  return (
    <div>
      <div className="eyebrow">{label}</div>
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className="text-[19px] font-semibold tracking-tight tnum" style={tone ? { color: TONE_VAR[tone] } : undefined}>
          {value}
        </span>
        {unit && <span className="text-[11px] text-muted">{unit}</span>}
      </div>
    </div>
  )
}

export function ChartTooltip({ active, payload, label, labelFormatter, valueFormatter }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-line-strong bg-surface-2/95 px-2.5 py-2 text-[11.5px] shadow-xl backdrop-blur">
      <div className="mb-1 font-medium text-fg-2">{labelFormatter ? labelFormatter(label) : label}</div>
      {payload
        .filter((p: any) => p.value !== undefined && p.value !== null)
        .map((p: any) => (
          <div key={p.dataKey ?? p.name} className="flex items-center justify-between gap-5">
            <span className="flex items-center gap-1.5 text-muted">
              <span className="h-2 w-2 rounded-full" style={{ background: p.color || p.stroke }} />
              {p.name}
            </span>
            <span className="font-medium tnum text-fg">{valueFormatter ? valueFormatter(p.value, p) : p.value}</span>
          </div>
        ))}
    </div>
  )
}
