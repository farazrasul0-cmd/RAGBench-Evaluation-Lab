import React from 'react'
import {
  Beaker,
  Layers,
  Sliders,
  BarChart2,
  FileSearch,
} from 'lucide-react'
import type { WorkbenchTab } from '../types'

interface HeaderProps {
  activeTab: WorkbenchTab
  onTabChange: (tab: WorkbenchTab) => void
}

export const Header: React.FC<HeaderProps> = ({ activeTab, onTabChange }) => {
  const tabs: { id: WorkbenchTab; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: 'overview', label: 'Overview', icon: Layers },
    { id: 'matrix', label: 'Matrix Builder', icon: Sliders },
    { id: 'compare', label: 'Comparative Dashboard', icon: BarChart2 },
    { id: 'trace', label: 'Trace Inspector', icon: FileSearch },
  ]

  return (
    <header className="px-6 py-3.5 bg-slate-900 border-b border-slate-800 text-white sticky top-0 z-50 backdrop-blur-md">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
            <Beaker className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-bold tracking-tight text-white">RAGBench</h1>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-mono font-medium">
                v1.0.0-PROD
              </span>
            </div>
            <div className="text-[11px] text-slate-400">Academic Evaluation Laboratory</div>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex items-center gap-1.5 p-1 bg-slate-950 rounded-xl border border-slate-800/80">
          {tabs.map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                data-testid={`tab-${tab.id}`}
                onClick={() => onTabChange(tab.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-sm font-semibold'
                    : 'text-slate-400 hover:text-white hover:bg-slate-900'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            )
          })}
        </nav>

        {/* Liveness Status */}
        <div className="hidden lg:flex items-center gap-2 text-xs text-slate-400">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span className="font-mono text-[11px] text-slate-300">Phase H Active</span>
        </div>
      </div>
    </header>
  )
}
