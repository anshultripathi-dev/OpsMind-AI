import { NavLink } from 'react-router-dom'
import { Activity, ShieldAlert, FlaskConical, Boxes, Sun, Moon, Github } from 'lucide-react'
import clsx from 'clsx'
import { Logo } from './Logo'
import { useLive } from '@/lib/store'

const NAV = [
  { to: '/', label: 'Live operations', icon: Activity, hint: 'G L' },
  { to: '/incidents', label: 'Incidents', icon: ShieldAlert, hint: 'G I' },
  { to: '/lab', label: 'Model lab', icon: FlaskConical, hint: 'G M' },
  { to: '/testbed', label: 'Testbed & data', icon: Boxes, hint: 'G T' },
]

export function Sidebar() {
  const { connected, status, theme, toggleTheme } = useLive()
  return (
    <aside className="sticky top-0 hidden h-screen w-[232px] shrink-0 flex-col border-r border-line bg-elev px-3 py-4 lg:flex">
      <div className="flex items-center gap-2.5 px-2 pb-5">
        <Logo />
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight">OpsMind AI</div>
          <div className="text-[10.5px] text-muted">Incident prediction & RCA</div>
        </div>
      </div>

      <nav className="flex flex-col gap-0.5">
        <div className="eyebrow px-2 pb-1.5">Workspace</div>
        {NAV.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              clsx(
                'group relative flex items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13px] font-medium transition-colors',
                isActive ? 'bg-accent-soft text-fg' : 'text-fg-2 hover:bg-surface-2 hover:text-fg',
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && <span className="absolute -left-3 top-1.5 bottom-1.5 w-[3px] rounded-r bg-accent" />}
                <Icon size={16} className={isActive ? 'text-accent' : 'text-muted group-hover:text-fg-2'} />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto space-y-3">
        <div className="card-flat px-3 py-2.5">
          <div className="flex items-center gap-2">
            <span className={clsx('h-2 w-2 rounded-full', connected ? 'bg-ok live-dot' : 'bg-crit')} />
            <span className="text-[12px] font-medium">{connected ? 'Pipeline connected' : 'Reconnecting…'}</span>
          </div>
          <div className="mt-1.5 grid grid-cols-2 gap-x-2 text-[11px] text-muted">
            <span>Source</span>
            <span className="text-right text-fg-2">{status?.mode === 'replay' ? 'Replay' : 'Simulator'}</span>
            <span>Inference</span>
            <span className="text-right tnum text-fg-2">{status ? `${status.stats.avg_ms.toFixed(1)} ms` : '—'}</span>
            <span>Windows</span>
            <span className="text-right tnum text-fg-2">{status?.stats.ticks ?? '—'}</span>
          </div>
        </div>
        <div className="flex items-center justify-between px-1">
          <span className="text-[11px] text-muted">v1.0 · ML course project</span>
          <button
            onClick={toggleTheme}
            className="grid h-7 w-7 place-items-center rounded-lg text-muted transition-colors hover:bg-surface-3 hover:text-fg"
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        </div>
      </div>
    </aside>
  )
}

export function MobileNav() {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-40 flex justify-around border-t border-line bg-elev/95 px-2 py-1.5 backdrop-blur lg:hidden">
      {NAV.map(({ to, label, icon: Icon }) => (
        <NavLink key={to} to={to} end={to === '/'} className={({ isActive }) => clsx('flex flex-col items-center gap-0.5 rounded-lg px-3 py-1 text-[10.5px]', isActive ? 'text-accent' : 'text-muted')}>
          <Icon size={18} />
          {label.split(' ')[0]}
        </NavLink>
      ))}
    </nav>
  )
}
