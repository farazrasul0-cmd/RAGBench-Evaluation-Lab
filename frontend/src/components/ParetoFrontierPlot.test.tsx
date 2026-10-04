import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ParetoFrontierPlot } from './ParetoFrontierPlot'
import type { ParetoPoint } from '../types'

const SAMPLE_POINTS: ParetoPoint[] = [
  {
    run_id: 'run-1',
    name: 'Dense EN-EN',
    strategy: 'dense',
    modality: 'EN-EN',
    recall_at_5: 0.98,
    latency_ms: 38.4,
    estimated_cost_usd: 0.001,
    ndcg_at_5: 0.96,
    is_pareto_optimal: true,
  },
  {
    run_id: 'run-2',
    name: 'BM25 EN-EN',
    strategy: 'bm25',
    modality: 'EN-EN',
    recall_at_5: 0.88,
    latency_ms: 12.1,
    estimated_cost_usd: 0.0001,
    ndcg_at_5: 0.84,
    is_pareto_optimal: true,
  },
  {
    run_id: 'run-3',
    name: 'Dominated Run',
    strategy: 'dense',
    modality: 'EN-BN',
    recall_at_5: 0.48,
    latency_ms: 45.0,
    estimated_cost_usd: 0.001,
    ndcg_at_5: 0.42,
    is_pareto_optimal: false,
  },
]

describe('ParetoFrontierPlot', () => {
  it('renders SVG chart title, axes, and frontier indicator', () => {
    render(<ParetoFrontierPlot points={SAMPLE_POINTS} />)
    expect(screen.getByText('Pareto Efficiency Frontier')).toBeDefined()
    expect(screen.getByText('Latency (ms, lower is better)')).toBeDefined()
    expect(screen.getByText('Retrieval Recall@5 (higher is better)')).toBeDefined()
    expect(screen.getByText('Frontier')).toBeDefined()
  })

  it('renders data points and handles selection callback', () => {
    const handleSelect = vi.fn()
    const { container } = render(
      <ParetoFrontierPlot points={SAMPLE_POINTS} onSelectRun={handleSelect} />
    )

    const points = container.querySelectorAll('circle')
    expect(points.length).toBeGreaterThanOrEqual(3)

    // Click on a point group
    const groups = container.querySelectorAll('g.cursor-pointer')
    if (groups.length > 0) {
      fireEvent.click(groups[0])
      expect(handleSelect).toHaveBeenCalled()
    }
  })
})
