import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { App } from './App'

describe('RAGBench Academic Workbench App', () => {
  it('renders application header, brand, and default overview view', () => {
    render(<App />)
    expect(screen.getByText('RAGBench')).toBeDefined()
    expect(screen.getByText('v1.0.0-PROD')).toBeDefined()
    expect(screen.getByText('Academic Research Workbench')).toBeDefined()
    expect(screen.getByText('Phase H Active')).toBeDefined()
  })

  it('navigates between all four workbench tabs via header', async () => {
    render(<App />)

    // Switch to Matrix Builder
    const matrixTab = screen.getByTestId('tab-matrix')
    fireEvent.click(matrixTab)
    expect(screen.getByText('Experiment Matrix Builder')).toBeDefined()

    // Switch to Comparative Dashboard
    const compareTab = screen.getByTestId('tab-compare')
    fireEvent.click(compareTab)
    expect(screen.getByText('Comparative Experiment Dashboard')).toBeDefined()

    // Switch to Trace Inspector
    const traceTab = screen.getByTestId('tab-trace')
    fireEvent.click(traceTab)
    expect(screen.getByText('Per-Question Trace & Hallucination Inspector')).toBeDefined()

    // Switch back to Overview
    const overviewTab = screen.getByTestId('tab-overview')
    fireEvent.click(overviewTab)
    expect(screen.getByText('Academic Research Workbench')).toBeDefined()
  })
})
