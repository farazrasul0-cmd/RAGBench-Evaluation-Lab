import React from 'react'
import { Header } from '../components/Header'
import { Layers, Cpu, Database, BarChart3 } from 'lucide-react'

export const DashboardView: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Header />
      <main className="flex-1 p-8 max-w-7xl mx-auto w-full">
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-2">Research Benchmark Dashboard</h2>
          <p className="text-slate-400">
            Systematic parameter sweeps across chunking, retrieval topologies, rerankers, and generation models.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 flex items-center gap-4">
            <Layers className="w-8 h-8 text-blue-400" />
            <div>
              <div className="text-xs text-slate-400">Chunking Engine</div>
              <div className="text-sm font-semibold text-white">4 Strategies</div>
            </div>
          </div>
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 flex items-center gap-4">
            <Database className="w-8 h-8 text-emerald-400" />
            <div>
              <div className="text-xs text-slate-400">Vector & Lexical</div>
              <div className="text-sm font-semibold text-white">Dense + BM25 (RRF)</div>
            </div>
          </div>
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 flex items-center gap-4">
            <Cpu className="w-8 h-8 text-purple-400" />
            <div>
              <div className="text-xs text-slate-400">Neural Reranker</div>
              <div className="text-sm font-semibold text-white">Cross-Encoder MS-Marco</div>
            </div>
          </div>
          <div className="p-4 rounded-lg bg-slate-900 border border-slate-800 flex items-center gap-4">
            <BarChart3 className="w-8 h-8 text-amber-400" />
            <div>
              <div className="text-xs text-slate-400">Verification Gate</div>
              <div className="text-sm font-semibold text-white">Phase 0 Initialized</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
