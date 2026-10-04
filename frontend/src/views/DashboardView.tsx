import React from 'react'
import {
  Layers,
  Cpu,
  Database,
  BarChart3,
  Sliders,
  BarChart2,
  FileSearch,
  ArrowRight,
  Sparkles,
  GitCommit,
} from 'lucide-react'
import type { WorkbenchTab } from '../types'

interface DashboardViewProps {
  onNavigate?: (tab: WorkbenchTab) => void
}

export const DashboardView: React.FC<DashboardViewProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-8">
      {/* Hero Banner */}
      <div className="p-6 md:p-8 rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border border-slate-800 relative overflow-hidden">
        <div className="max-w-2xl relative z-10 space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-mono">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Phase G Formally Approved · Commit eb3bb4d</span>
          </div>
          <h2 className="text-3xl font-extrabold text-white tracking-tight">
            Academic Research Workbench
          </h2>
          <p className="text-sm text-slate-300 leading-relaxed">
            Multi-dimensional evaluation system for Retrieval-Augmented Generation.
            Systematic parameter sweeps across chunking strategies, embedding dimensions,
            retrieval topologies, and neural rerankers with statistical significance verification.
          </p>
          <div className="flex flex-wrap gap-3 pt-2">
            <button
              onClick={() => onNavigate?.('compare')}
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold flex items-center gap-2 transition-colors shadow-sm"
            >
              <BarChart2 className="w-4 h-4" /> Open Comparative Dashboard
            </button>
            <button
              onClick={() => onNavigate?.('matrix')}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold flex items-center gap-2 border border-slate-700 transition-colors"
            >
              <Sliders className="w-4 h-4" /> Experiment Matrix Builder
            </button>
          </div>
        </div>
      </div>

      {/* Core Architectural Pillars */}
      <div>
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Evaluation Engines & Topologies
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center gap-4">
            <div className="p-2.5 rounded-lg bg-blue-500/10 text-blue-400">
              <Layers className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs text-slate-400 font-medium">Chunking Engine</div>
              <div className="text-sm font-bold text-white">4 Strategies</div>
              <div className="text-[11px] text-slate-500 font-mono mt-0.5">Fixed · Sentence · Recursive</div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center gap-4">
            <div className="p-2.5 rounded-lg bg-emerald-500/10 text-emerald-400">
              <Database className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs text-slate-400 font-medium">Vector & Lexical</div>
              <div className="text-sm font-bold text-white">Dense + BM25</div>
              <div className="text-[11px] text-slate-500 font-mono mt-0.5">BGE-M3 (1024-dim) + RRF</div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center gap-4">
            <div className="p-2.5 rounded-lg bg-purple-500/10 text-purple-400">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs text-slate-400 font-medium">Neural Reranker</div>
              <div className="text-sm font-bold text-white">MS-Marco Cross-Enc.</div>
              <div className="text-[11px] text-slate-500 font-mono mt-0.5">Two-Stage Pipeline</div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 flex items-center gap-4">
            <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-400">
              <BarChart3 className="w-6 h-6" />
            </div>
            <div>
              <div className="text-xs text-slate-400 font-medium">Statistical Rigor</div>
              <div className="text-sm font-bold text-white">Wilcoxon & Holm</div>
              <div className="text-[11px] text-slate-500 font-mono mt-0.5">m=4 Hypothesis Family</div>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Exploration Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div
          onClick={() => onNavigate?.('matrix')}
          className="p-5 rounded-xl bg-slate-900 border border-slate-800 hover:border-indigo-500/50 cursor-pointer transition-all group flex flex-col justify-between"
        >
          <div>
            <div className="w-10 h-10 rounded-lg bg-indigo-500/10 text-indigo-400 flex items-center justify-center mb-4">
              <Sliders className="w-5 h-5" />
            </div>
            <h4 className="text-base font-bold text-white group-hover:text-indigo-300 transition-colors">
              Matrix Sweep Builder
            </h4>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Design multi-dimensional combinatorial experiments with live point calculations, complexity categorization, and YAML export.
            </p>
          </div>
          <div className="mt-4 flex items-center text-xs font-semibold text-indigo-400 group-hover:translate-x-1 transition-transform">
            <span>Configure Matrix</span> <ArrowRight className="w-3.5 h-3.5 ml-1" />
          </div>
        </div>

        <div
          onClick={() => onNavigate?.('compare')}
          className="p-5 rounded-xl bg-slate-900 border border-slate-800 hover:border-indigo-500/50 cursor-pointer transition-all group flex flex-col justify-between"
        >
          <div>
            <div className="w-10 h-10 rounded-lg bg-purple-500/10 text-purple-400 flex items-center justify-center mb-4">
              <BarChart2 className="w-5 h-5" />
            </div>
            <h4 className="text-base font-bold text-white group-hover:text-purple-300 transition-colors">
              Comparative Dashboard
            </h4>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Explore pure SVG Pareto frontier trade-offs and 5-axis radar chart overlays comparing 12 empirical reference sweeps.
            </p>
          </div>
          <div className="mt-4 flex items-center text-xs font-semibold text-purple-400 group-hover:translate-x-1 transition-transform">
            <span>View Comparative Sweep</span> <ArrowRight className="w-3.5 h-3.5 ml-1" />
          </div>
        </div>

        <div
          onClick={() => onNavigate?.('trace')}
          className="p-5 rounded-xl bg-slate-900 border border-slate-800 hover:border-indigo-500/50 cursor-pointer transition-all group flex flex-col justify-between"
        >
          <div>
            <div className="w-10 h-10 rounded-lg bg-teal-500/10 text-teal-400 flex items-center justify-center mb-4">
              <FileSearch className="w-5 h-5" />
            </div>
            <h4 className="text-base font-bold text-white group-hover:text-teal-300 transition-colors">
              Trace & Hallucination Inspector
            </h4>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              Audit sentence-level generation outputs with color-coded entailing vs. hallucinated spans and interactive citation jumps.
            </p>
          </div>
          <div className="mt-4 flex items-center text-xs font-semibold text-teal-400 group-hover:translate-x-1 transition-transform">
            <span>Inspect Queries</span> <ArrowRight className="w-3.5 h-3.5 ml-1" />
          </div>
        </div>
      </div>

      {/* Academic Citation & Baseline Box */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <GitCommit className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>
            Locked Empirical Baseline: <span className="text-white font-mono font-semibold">eb3bb4d</span> (BAAI/bge-m3, 1024 dims, N=25 matched units)
          </span>
        </div>
        <div className="font-mono text-[11px] text-slate-500">
          RAGBench Academic Evaluation Laboratory v1.0.0
        </div>
      </div>
    </div>
  )
}
