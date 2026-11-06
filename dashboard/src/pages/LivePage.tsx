import { useLive } from '@/lib/store'
import { StatusHero } from '@/components/live/StatusHero'
import { KpiRow } from '@/components/live/KpiRow'
import { Topology } from '@/components/live/Topology'
import { RootCausePanel } from '@/components/live/RootCausePanel'
import { MetricExplorer, RiskTimeline } from '@/components/live/Charts'
import { ChaosConsole } from '@/components/live/ChaosConsole'
import { ClassifierCard, EventFeed, LogStream } from '@/components/live/Feeds'

export function LivePage() {
  const ready = useLive((s) => s.ready)
  if (!ready)
    return (
      <div className="space-y-4">
        <div className="card skeleton h-[104px]" />
        <div className="grid grid-cols-4 gap-4">{[0, 1, 2, 3].map((i) => <div key={i} className="card skeleton h-[150px]" />)}</div>
        <div className="card skeleton h-[420px]" />
      </div>
    )
  return (
    <div className="space-y-4">
      <StatusHero />
      <KpiRow />
      <div className="grid gap-4 xl:grid-cols-12">
        <div className="xl:col-span-8"><Topology /></div>
        <div className="xl:col-span-4"><RootCausePanel /></div>
      </div>
      <div className="grid gap-4 xl:grid-cols-12">
        <div className="space-y-4 xl:col-span-8">
          <RiskTimeline />
          <MetricExplorer />
        </div>
        <div className="space-y-4 xl:col-span-4">
          <ChaosConsole />
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-12">
        <div className="xl:col-span-4"><EventFeed /></div>
        <div className="xl:col-span-5"><LogStream /></div>
        <div className="xl:col-span-3"><ClassifierCard /></div>
      </div>
    </div>
  )
}
