/**
 * RAGBench REST API Client
 */

export interface SystemHealth {
  status: string
  project: string
  version: string
}

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch('/api/v1/health')
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.statusText}`)
  }
  return res.json() as Promise<SystemHealth>
}
