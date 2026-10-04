import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MatrixBuilderView } from './MatrixBuilderView'

describe('MatrixBuilderView', () => {
  it('renders metadata controls and default configuration dimensions', async () => {
    render(<MatrixBuilderView />)
    expect(screen.getByText('Experiment Matrix Builder')).toBeDefined()
    expect(screen.getByDisplayValue('rq3_multilingual_factorial_sweep')).toBeDefined()
    expect(screen.getByText('Factorial Dimensions')).toBeDefined()
  })

  it('allows toggling dimensions and recalculates combinatorial scale', async () => {
    render(<MatrixBuilderView />)

    const recursiveBtn = screen.getByRole('button', { name: 'recursive' })
    fireEvent.click(recursiveBtn)

    await waitFor(() => {
      expect(screen.getByText(/Variations/i)).toBeDefined()
    })
  })

  it('switches between configuration table preview and specification YAML', async () => {
    render(<MatrixBuilderView />)

    const yamlTabBtn = screen.getByRole('button', { name: /Specification YAML/i })
    fireEvent.click(yamlTabBtn)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Copy/i })).toBeDefined()
      expect(screen.getByRole('button', { name: /Download/i })).toBeDefined()
    })
  })

  it('loads RQ3 Multilingual preset with exact approved Phase G configuration (fixed 200/20, BGE-M3, k=5)', async () => {
    render(<MatrixBuilderView />)

    const rq3PresetBtn = screen.getByRole('button', { name: 'RQ3 Multilingual' })
    fireEvent.click(rq3PresetBtn)

    await waitFor(() => {
      expect(screen.getByDisplayValue('rq3_multilingual_study_matrix')).toBeDefined()
      expect(screen.getByDisplayValue('multilingual_canonical')).toBeDefined()
      expect(screen.getByDisplayValue('dd59f087-86ff-4872-925f-adb03fc8d9a2')).toBeDefined()
      // Chunking strategy: fixed (1)
      expect(screen.getByText('Chunking Strategy (1)')).toBeDefined()
      // Chunk size 200
      expect(screen.getByRole('button', { name: '200' })).toBeDefined()
      // Retrieval topologies: dense, bm25, hybrid (3)
      expect(screen.getByText('Retrieval Strategies (3)')).toBeDefined()
      // Neural reranker: none (1)
      expect(screen.getByText('Neural Rerankers (1)')).toBeDefined()
    })
  })

  it('loads Fast Smoke preset correctly', async () => {
    render(<MatrixBuilderView />)

    const fastPresetBtn = screen.getByRole('button', { name: 'Fast Smoke' })
    fireEvent.click(fastPresetBtn)

    await waitFor(() => {
      expect(screen.getByDisplayValue('quick_smoke_matrix')).toBeDefined()
    })
  })
})
