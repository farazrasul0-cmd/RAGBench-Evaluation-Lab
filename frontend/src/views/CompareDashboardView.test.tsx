import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { CompareDashboardView } from './CompareDashboardView'

describe('CompareDashboardView', () => {
  it('renders KPI summary cards including accurate RQ3 empirical findings', async () => {
    render(<CompareDashboardView />)
    expect(screen.getByText('Comparative Experiment Dashboard')).toBeDefined()
    expect(screen.getByText('Monolingual Peak (EN)')).toBeDefined()
    expect(screen.getByText('RQ3 Hypothesis Family')).toBeDefined()
    expect(screen.getByText('H1 Supported')).toBeDefined()
    expect(screen.getByText('CONTROLLED COMPARISON')).toBeDefined()

    await waitFor(() => {
      expect(screen.getByText('Full Experimental Run Matrix')).toBeDefined()
      expect(screen.getByText('Dense (BGE-M3) x EN-EN')).toBeDefined()
    })
  })

  it('filters runs by modality and topology', async () => {
    render(<CompareDashboardView />)

    await waitFor(() => {
      expect(screen.getByText('Dense (BGE-M3) x EN-EN')).toBeDefined()
    })

    // Filter by BN-BN
    const bnbnBtn = screen.getByRole('button', { name: 'BN-BN' })
    fireEvent.click(bnbnBtn)

    await waitFor(() => {
      expect(screen.getByText('Dense (BGE-M3) x BN-BN')).toBeDefined()
    })
  })
})
