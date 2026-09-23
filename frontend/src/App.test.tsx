import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import App from './App'

describe('RAGBench Frontend Scaffold', () => {
  it('renders application header and dashboard view', () => {
    render(<App />)
    expect(screen.getByText('RAGBench')).toBeDefined()
    expect(screen.getByText('Research Benchmark Dashboard')).toBeDefined()
    expect(screen.getByText('Phase 0 Initialized')).toBeDefined()
  })
})
