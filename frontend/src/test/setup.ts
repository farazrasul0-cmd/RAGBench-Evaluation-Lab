import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock scrollIntoView for JSDOM
window.HTMLElement.prototype.scrollIntoView = vi.fn()
