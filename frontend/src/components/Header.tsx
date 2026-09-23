import React from 'react'
import { Activity, Beaker } from 'lucide-react'

export const Header: React.FC = () => {
  return (
    <header className="flex items-center justify-between px-6 py-4 bg-slate-900 border-b border-slate-800 text-white">
      <div className="flex items-center gap-3">
        <Beaker className="w-6 h-6 text-indigo-400" />
        <h1 className="text-xl font-bold tracking-tight">RAGBench</h1>
        <span className="text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono">
          v1.0.0-PROD
        </span>
      </div>
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <Activity className="w-4 h-4 text-emerald-400" />
        <span>Evaluation Laboratory Active</span>
      </div>
    </header>
  )
}
