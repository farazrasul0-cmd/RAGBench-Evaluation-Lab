import React, { useState } from 'react'
import type { WorkbenchTab } from './types'
import { Header } from './components/Header'
import { DashboardView } from './views/DashboardView'
import { MatrixBuilderView } from './views/MatrixBuilderView'
import { CompareDashboardView } from './views/CompareDashboardView'
import { TraceInspectorView } from './views/TraceInspectorView'

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<WorkbenchTab>('overview')

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />
      <main className="flex-1 p-6 md:p-8 max-w-7xl mx-auto w-full">
        {activeTab === 'overview' && <DashboardView onNavigate={setActiveTab} />}
        {activeTab === 'matrix' && <MatrixBuilderView />}
        {activeTab === 'compare' && <CompareDashboardView />}
        {activeTab === 'trace' && <TraceInspectorView />}
      </main>
      <footer className="border-t border-slate-800/80 py-4 px-6 bg-slate-950 text-center text-xs text-slate-500 font-mono">
        RAGBench Academic Workbench · Immutable Baseline @ eb3bb4d · Real BGE-M3 Inference
      </footer>
    </div>
  )
}

export default App
