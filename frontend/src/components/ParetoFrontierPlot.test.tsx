import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ParetoFrontierPlot } from './ParetoFrontierPlot'
import { computeParetoFrontier, validateControlledComparison } from '../api/client'
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
    control_signature: {
      dataset_id: 'multilingual_canonical',
      dataset_version_id: 'v1.0',
      query_population: 'N=25',
      top_k: 5,
      protocol_version: 'v1.0',
    },
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
    control_signature: {
      dataset_id: 'multilingual_canonical',
      dataset_version_id: 'v1.0',
      query_population: 'N=25',
      top_k: 5,
      protocol_version: 'v1.0',
    },
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
    control_signature: {
      dataset_id: 'multilingual_canonical',
      dataset_version_id: 'v1.0',
      query_population: 'N=25',
      top_k: 5,
      protocol_version: 'v1.0',
    },
  },
]

describe('ParetoFrontierPlot', () => {
  it('renders SVG chart title, axes, and controlled validation badge', () => {
    render(<ParetoFrontierPlot points={SAMPLE_POINTS} />)
    expect(screen.getByText('Pareto Efficiency Frontier')).toBeDefined()
    expect(screen.getByText('Controlled Validated')).toBeDefined()
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

    const groups = container.querySelectorAll('g.cursor-pointer')
    if (groups.length > 0) {
      fireEvent.click(groups[0])
      expect(handleSelect).toHaveBeenCalled()
    }
  })

  it('detects divergent control dimensions and suppresses frontier calculation', () => {
    const divergentPoints: ParetoPoint[] = [
      {
        run_id: 'run-k5',
        name: 'Run K=5',
        strategy: 'dense',
        recall_at_5: 0.90,
        latency_ms: 20,
        estimated_cost_usd: 0,
        ndcg_at_5: 0,
        is_pareto_optimal: true,
        control_signature: {
          dataset_id: 'ds1',
          dataset_version_id: 'v1',
          query_population: 'N=25',
          top_k: 5,
          protocol_version: 'v1.0',
        },
      },
      {
        run_id: 'run-k10',
        name: 'Run K=10',
        strategy: 'dense',
        recall_at_5: 0.95,
        latency_ms: 30,
        estimated_cost_usd: 0,
        ndcg_at_5: 0,
        is_pareto_optimal: true,
        control_signature: {
          dataset_id: 'ds1',
          dataset_version_id: 'v1',
          query_population: 'N=25',
          top_k: 10, // Divergent top_k!
          protocol_version: 'v1.0',
        },
      },
    ]

    render(<ParetoFrontierPlot points={divergentPoints} />)
    expect(screen.getByText('Incomparable Conditions')).toBeDefined()
    expect(screen.getByText(/Incomparable Evaluation Conditions: Selected sweeps differ in \[top_k\]/i)).toBeDefined()
  })

  it('verifies validateControlledComparison directly', () => {
    const res = validateControlledComparison(SAMPLE_POINTS)
    expect(res.is_controlled).toBe(true)
    expect(res.divergent_fields.length).toBe(0)
  })

  it('verifies computeParetoFrontier dominance math correctly', () => {
    const raw: ParetoPoint[] = [
      { run_id: 'fast_good', name: 'A', strategy: 'dense', recall_at_5: 0.9, latency_ms: 10, estimated_cost_usd: 0, ndcg_at_5: 0, is_pareto_optimal: false },
      { run_id: 'slow_worse', name: 'B', strategy: 'dense', recall_at_5: 0.8, latency_ms: 20, estimated_cost_usd: 0, ndcg_at_5: 0, is_pareto_optimal: false },
      { run_id: 'equal_lat_worse_rec', name: 'C', strategy: 'dense', recall_at_5: 0.7, latency_ms: 10, estimated_cost_usd: 0, ndcg_at_5: 0, is_pareto_optimal: false },
      { run_id: 'equal_rec_worse_lat', name: 'D', strategy: 'dense', recall_at_5: 0.9, latency_ms: 15, estimated_cost_usd: 0, ndcg_at_5: 0, is_pareto_optimal: false },
    ]

    const evaluated = computeParetoFrontier(raw)
    const optMap = Object.fromEntries(evaluated.map((p) => [p.run_id, p.is_pareto_optimal]))

    expect(optMap['fast_good']).toBe(true)
    expect(optMap['slow_worse']).toBe(false)
    expect(optMap['equal_lat_worse_rec']).toBe(false)
    expect(optMap['equal_rec_worse_lat']).toBe(false)
  })
})
