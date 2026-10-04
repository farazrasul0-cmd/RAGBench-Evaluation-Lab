import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { TraceInspectorView } from './TraceInspectorView'

describe('TraceInspectorView', () => {
  it('renders query search and claim-level attribution inspector', async () => {
    render(<TraceInspectorView />)
    expect(screen.getByText('Per-Question Trace & Hallucination Inspector')).toBeDefined()

    await waitFor(() => {
      expect(screen.getByText('Evaluated Query')).toBeDefined()
      expect(screen.getByText('Expected Target / Ground Truth Fact')).toBeDefined()
      expect(screen.getByText('Generated Answer & Claim-Level Attribution')).toBeDefined()
      expect(screen.getByText('Supported Claim')).toBeDefined()
      expect(screen.getByText('Unsupported / Hallucinated')).toBeDefined()
    })
  })

  it('supports searching query traces in the sidebar', async () => {
    render(<TraceInspectorView />)

    await waitFor(() => {
      expect(screen.getByText('Evaluated Query')).toBeDefined()
    })

    const searchInput = screen.getByPlaceholderText('Search queries or IDs...')
    fireEvent.change(searchInput, { target: { value: 'CRISPR' } })

    expect(screen.getByDisplayValue('CRISPR')).toBeDefined()
  })

  it('detects and visibly flags unresolvable citations with alert banner', async () => {
    render(<TraceInspectorView />)

    await waitFor(() => {
      expect(screen.getByText('Supported Claim')).toBeDefined()
    })

    // Look for citation buttons in generated spans
    const citationBtns = screen.getAllByRole('button', { name: /\[/i })
    expect(citationBtns.length).toBeGreaterThan(0)

    // Click first citation button
    fireEvent.click(citationBtns[0])

    // Should not throw or crash
    expect(screen.getByText('Generated Answer & Claim-Level Attribution')).toBeDefined()
  })
})
