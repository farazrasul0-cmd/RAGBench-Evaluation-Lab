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
      recall_at_10: 0.98,
      precision_at_5: 0.20,
      faithfulness: 0.96,
      citation_accuracy: 0.94,
      cost_efficiency: 0.75,
    },
  },
  {
    run_id: 'run-2',
    name: 'BM25 Lexical',
    color: '#f59e0b',
    metrics: {
      recall_at_10: 0.88,
      precision_at_5: 0.18,
      faithfulness: 0.89,
      citation_accuracy: 0.86,
      cost_efficiency: 0.98,
    },
  },
]

describe('RadarChart', () => {
  it('renders 5-axis pentagon labels and chart title', () => {
    render(<RadarChart data={SAMPLE_RADAR} />)
    expect(screen.getByText('Multi-Dimensional Metric Radar')).toBeDefined()
    expect(screen.getByText('Recall@10')).toBeDefined()
    expect(screen.getByText('Precision@5')).toBeDefined()
    expect(screen.getByText('Faithfulness')).toBeDefined()
    expect(screen.getByText('Citation Acc.')).toBeDefined()
    expect(screen.getByText('Cost Efficiency')).toBeDefined()
  })

  it('renders series buttons and supports toggle callbacks', () => {
    const handleToggle = vi.fn()
    render(<RadarChart data={SAMPLE_RADAR} onToggleRun={handleToggle} />)

    const toggleBtn = screen.getByRole('button', { name: /Dense \(BGE-M3\)/i })
    fireEvent.click(toggleBtn)
    expect(handleToggle).toHaveBeenCalledWith('run-1')
  })
})
