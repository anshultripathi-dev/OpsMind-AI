export type ComponentId = 'gateway' | 'auth' | 'order' | 'payment' | 'postgres' | 'redis'
export type FaultType =
  | 'cpu_saturation'
  | 'memory_leak'
  | 'db_slowdown'
  | 'network_latency'
  | 'service_crash'
  | 'cache_failure'

export interface Driver {
  feature: string
  label: string
  shap: number
  z: number
}
export interface RcaEntry {
  component: ComponentId
  score: number
  evidence: number
  drivers: Driver[]
}
export interface IncidentRec {
  id: string
  started_at: number
  detected_at: number
  warning_at: number | null
  baseline_at: number | null
  impact_at: number | null
  resolved_at: number | null
  peak_risk: number
  type: FaultType | null
  type_name: string | null
  rca: { component: ComponentId; score: number }[] | null
  status: 'open' | 'resolved'
  lead_s: number | null
  baseline_lead_s: number | null
}
export interface EventRec {
  t: number
  kind: 'early_warning' | 'detected' | 'threshold_alert' | 'impact' | 'resolved' | 'injected' | 'cleared' | 'novel_pattern'
  severity: 'info' | 'warn' | 'critical'
  title: string
  detail: string
}
export interface TruthFault {
  type: FaultType
  target: ComponentId
  intensity: number
  severity: number
  start?: number
  end?: number
}
export interface Tick {
  type: 'tick'
  i: number
  t: number
  sample: Record<ComponentId, Record<string, number>>
  deviation: Record<ComponentId, number>
  anomaly: { score: number; threshold: number; ratio: number; flagged: boolean; sustained: boolean }
  classifier: {
    label: string
    name: string
    confidence: number
    probs: Record<string, number>
    likely_fault: FaultType
    likely_fault_name: string
  }
  risk: { prob: number; threshold: number; alert: boolean }
  baseline: { firing: { rule: string; component: ComponentId; value: number; threshold: number }[]; alert: boolean }
  rca: RcaEntry[] | null
  pattern: { id: number; name: string; signature?: string; dominant?: string; purity?: number; distance: number; novel: boolean }
  slo: { impacted: boolean; err: number; lat: number }
  logs: { t: number; service: ComponentId; level: string; template_id: number; msg: string }[]
  incident: { current: IncidentRec | null; history: IncidentRec[] }
  truth: { faults: TruthFault[]; spike: number; batch: boolean } | null
  events: EventRec[]
}
export interface SessionStatus {
  type: 'status'
  mode: 'sim' | 'replay'
  models?: 'simulator' | 'real'
  paused: boolean
  speed: number
  replay_done: boolean
  scenario: string | null
  source: { kind: string; run_id?: string; progress?: number; truth?: Record<string, unknown> }
  stats: { ticks: number; avg_ms: number; max_ms: number }
  scenarios: { id: string; title: string; description: string }[]
  clients: number
}
export interface Config {
  components: { id: ComponentId; label: string; kind: string; metrics: string[]; calls: ComponentId[] }[]
  edges: { source: ComponentId; target: ComponentId; fan: number }[]
  metrics: Record<string, { label: string; unit: string }>
  faults: Record<FaultType, { label: string; tool: string; targets: ComponentId[]; primary: ComponentId | null }>
  classes: string[]
  thresholds: Record<string, { metric: string; op: string; value: number }>
  slo: { err_rate: number; latency_ms: number }
  scenarios: { id: string; title: string; description: string }[]
}
