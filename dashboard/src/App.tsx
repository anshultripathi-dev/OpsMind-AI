import { useEffect } from 'react'
import { BrowserRouter, Route, Routes, useNavigate } from 'react-router-dom'
import { Sidebar, MobileNav } from '@/components/layout/Sidebar'
import { Topbar } from '@/components/layout/Topbar'
import { LivePage } from '@/pages/LivePage'
import { IncidentsPage } from '@/pages/IncidentsPage'
import { ModelLabPage } from '@/pages/ModelLabPage'
import { TestbedPage } from '@/pages/TestbedPage'
import { useLive } from '@/lib/store'

function Shortcuts() {
  const nav = useNavigate()
  useEffect(() => {
    let g = false
    const h = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input,textarea,select')) return
      if (g) {
        const m: Record<string, string> = { l: '/', i: '/incidents', m: '/lab', t: '/testbed' }
        if (m[e.key]) nav(m[e.key])
        g = false
        return
      }
      if (e.key === 'g') {
        g = true
        setTimeout(() => (g = false), 900)
      }
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [nav])
  return null
}

export default function App() {
  const connect = useLive((s) => s.connect)
  useEffect(() => connect(), [connect])
  return (
    <BrowserRouter>
      <Shortcuts />
      <div className="flex min-h-screen">
        <Sidebar />
        <div className="min-w-0 flex-1 pb-16 lg:pb-0">
          <Topbar />
          <main className="mx-auto w-full max-w-[1760px] px-5 py-5">
            <Routes>
              <Route path="/" element={<LivePage />} />
              <Route path="/incidents" element={<IncidentsPage />} />
              <Route path="/lab" element={<ModelLabPage />} />
              <Route path="/testbed" element={<TestbedPage />} />
            </Routes>
          </main>
        </div>
      </div>
      <MobileNav />
    </BrowserRouter>
  )
}
