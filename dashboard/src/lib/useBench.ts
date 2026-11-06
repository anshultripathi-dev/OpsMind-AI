import { useEffect, useState } from 'react'
import { get } from './api'

/* eslint-disable @typescript-eslint/no-explicit-any */
const bench: Record<string, any> = {}
let dataset: any = null

export function useBench(source: 'simulator' | 'real' = 'simulator'): any {
  const [d, setD] = useState<any>(bench[source] ?? null)
  useEffect(() => {
    if (bench[source]) { setD(bench[source]); return }
    setD(null)
    get(`/api/benchmarks?source=${source}`).then((b) => { bench[source] = b; setD(b) }).catch(() => setD(null))
  }, [source])
  return d
}

export function useSources(): { simulator: boolean; real: boolean } {
  const [s, setS] = useState({ simulator: true, real: false })
  useEffect(() => { get<{ simulator: boolean; real: boolean }>('/api/benchmarks/sources').then(setS).catch(() => {}) }, [])
  return s
}

export function useDataset(): any {
  const [d, setD] = useState<any>(dataset)
  useEffect(() => {
    if (!dataset) get('/api/dataset').then((b) => { dataset = b; setD(b) }).catch(() => {})
  }, [])
  return d
}
