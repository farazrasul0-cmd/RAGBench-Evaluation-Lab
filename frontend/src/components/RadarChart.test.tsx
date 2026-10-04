import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { RadarChart } from './RadarChart'
import type { RadarMetricData } from '../types'

const SAMPLE_RADAR: RadarMetricData[] = [
  {
    run_id: 'run-1',
    name: 'Dense (BGE-M3)',
    color: '#3b82f6',
    metrics: {
      recall_at_5: 0.98,
      precision_at_5: 0.20,
      mrr_at_5: 0.98,
      ndcg_at_5: 0.96,
      faithfulness: 0.96,
    },
  },
  {
    run_id: 'run-2',
    name: 'BM25 Lexical',
    color: '#f59e0b',
    metrics: {
      recall_at_5: 0.88,
      precision_at_5: 0.18,
      mrr_at_5: 0.88,
      ndcg_at_5: 0.84,
      faithfulness: 0.98,
    },
  },
]

describe('RadarChart', () => {
  it('renders 5 authoritative backend IR and generation axis labels', () => {
    render(<RadarChart data={SAMPLE_RADAR} />)
    expect(screen.getByText('Multi-Dimensional Metric Radar')).toBeDefined()
    expect(screen.getByText('Recall@5')).toBeDefined()
    expect(screen.getByText('Precision@5')).toBeDefined()
    expect(screen.getByText('MRR@5')).toBeDefined()
    expect(screen.getByText('NDCG@5')).toBeDefined()
    expect(screen.getByText('Faithfulness')).toBeDefined()
  })

  it('renders series buttons and supports toggle callbacks', () => {
    const handleToggle = vi.fn()
    render(<RadarChart data={SAMPLE_RADAR} onToggleRun={handleToggle} />)

    const toggleBtn = screen.getByRole('button', { name: /Dense \(BGE-M3\)/i })
    fireEvent.click(toggleBtn)
    expect(handleToggle).toHaveBeenCalledWith('run-1')
  })

  it('handles zero and missing metrics gracefully without NaN coordinates', () => {
    const edgeData: RadarMetricData[] = [
      {
        run_id: 'zero_run',
        name: 'Zero Metric Run',
        color: '#ef4444',
        metrics: {
          recall_at_5: 0.0,
          precision_at_5: 0.0,
          mrr_at_5: 0.0,
          ndcg_at_5: 0.0,
          faithfulness: 0.0,
        },
      },
    ]

    const { container } = render(<RadarChart data={edgeData} />)
    const polygon = container.querySelector('polygon[fill="#ef4444"]')
    expect(polygon).toBeDefined()
    const pointsAttr = polygon?.getAttribute('points')
    expect(pointsAttr).not.toContain('NaN')
  })

  it('renders up to 5 concurrent series without rendering errors', () => {
    const multiSeries: RadarMetricData[] = [1, 2, 3, 4, 5].map((i) => ({
      run_id: `series-${i}`,
      name: `Series ${i}`,
      color: `#${i}${i}4488`,
      metrics: {
        recall_at_5: i * 0.2,
        precision_at_5: i * 0.2,
        mrr_at_5: i * 0.2,
        ndcg_at_5: i * 0.2,
        faithfulness: i * 0.2,
      },
    }))

    const { container } = render(<RadarChart data={multiSeries} />)
    expect(container.querySelectorAll('polygon').length).toBeGreaterThanOrEqual(5)
  })
})
