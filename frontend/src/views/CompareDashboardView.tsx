import React, { useState, useEffect, useMemo } from 'react'
import {
  BarChart2,
  TrendingUp,
  Clock,
  Layers,
  Award,
  Filter,
  CheckCircle2,
  Sparkles,
} from 'lucide-react'
import type { ComparisonRunSummary, ParetoPoint, RadarMetricData } from '../types'
import {
  fetchRunsForComparison,
  fetchParetoData,
  fetchRadarData,
} from '../api/client'
import { ParetoFrontierPlot } from '../components/ParetoFrontierPlot'
import { RadarChart } from '../components/RadarChart'

export const CompareDashboardView: React.FC = () => {
  const [runs, setRuns] = useState<ComparisonRunSummary[]>([])
  const [paretoPoints, setParetoPoints] = useState<ParetoPoint[]>([])
  const [radarData, setRadarData] = useState<RadarMetricData[]>([])
  
  // Filter states
  const [selectedModality, setSelectedModality] = useState<string>('ALL')
  const [selectedStrategy, setSelectedStrategy] = useState<string>('ALL')
  const [sortField, setSortField] = useState<keyof ComparisonRunSummary>('recall_at_5')
  const [sortAsc, setSortAsc] = useState<boolean>(false)

  // Selected run IDs for radar chart comparison
  const [activeRadarIds, setActiveRadarIds] = useState<string[]>([])
  const [highlightedRunId, setHighlightedRunId] = useState<string | undefined>()

  useEffect(() => {
    let isMounted = true
    
    Promise.all([
      fetchRunsForComparison(),
      fetchParetoData(),
      fetchRadarData(),
    ])
      .then(([runsRes, paretoRes, radarRes]) => {
        if (isMounted) {
          setRuns(runsRes)
          setParetoPoints(paretoRes)
          setRadarData(radarRes)

          // Default radar selection: top 3 reference runs
          if (radarRes.length > 0) {
            setActiveRadarIds(radarRes.slice(0, 3).map((r) => r.run_id))
          }
                  }
      })
      .catch(() => {})

    return () => {
      isMounted = false
    }
  }, [])

  const toggleRadarRun = (runId: string) => {
    if (activeRadarIds.includes(runId)) {
      if (activeRadarIds.length > 1) {
        setActiveRadarIds(activeRadarIds.filter((id) => id !== runId))
      }
    } else {
      if (activeRadarIds.length < 5) {
        setActiveRadarIds([...activeRadarIds, runId])
      }
    }
  }

  // Filtered runs
  const filteredRuns = useMemo(() => {
    return runs.filter((r) => {
      const matchMod = selectedModality === 'ALL' || r.modality === selectedModality
      const matchStrat = selectedStrategy === 'ALL' || r.strategy.toLowerCase() === selectedStrategy.toLowerCase()
      return matchMod && matchStrat
    }).sort((a, b) => {
      const valA = a[sortField] ?? 0
      const valB = b[sortField] ?? 0
      if (typeof valA === 'number' && typeof valB === 'number') {
        return sortAsc ? valA - valB : valB - valA
      }
      return sortAsc
        ? String(valA).localeCompare(String(valB))
        : String(valB).localeCompare(String(valA))
    })
  }, [runs, selectedModality, selectedStrategy, sortField, sortAsc])

  // Filtered pareto points matching current modality filter
  const filteredParetoPoints = useMemo(() => {
    return paretoPoints.filter((pt) => {
      const matchMod = selectedModality === 'ALL' || pt.modality === selectedModality
      const matchStrat = selectedStrategy === 'ALL' || pt.strategy.toLowerCase() === selectedStrategy.toLowerCase()
      return matchMod && matchStrat
    })
  }, [paretoPoints, selectedModality, selectedStrategy])

  // Helper formatting badges
  const getModalityBadge = (modality: string) => {
    switch (modality) {
      case 'EN-EN':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30'
      case 'BN-BN':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
      case 'EN-BN':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/30'
      case 'BN-EN':
        return 'bg-rose-500/20 text-rose-300 border-rose-500/30'
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700'
    }
  }

  const getStrategyBadge = (strat: string) => {
    switch (strat.toLowerCase()) {
      case 'dense':
        return 'bg-blue-600/20 text-blue-300 border-blue-600/40'
      case 'bm25':
        return 'bg-amber-600/20 text-amber-300 border-amber-600/40'
      case 'hybrid':
        return 'bg-purple-600/20 text-purple-300 border-purple-600/40'
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700'
    }
  }

  const handleSort = (field: keyof ComparisonRunSummary) => {
    if (sortField === field) {
      setSortAsc(!sortAsc)
    } else {
      setSortField(field)
      setSortAsc(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <BarChart2 className="w-5 h-5 text-indigo-400" />
            Comparative Experiment Dashboard
          </h2>
          <p className="text-sm text-slate-400">
            Empirical evaluation of 12 reference sweeps across retrieval topologies and monolingual vs. cross-lingual modalities.
          </p>
        </div>

        {/* Global Stats Counter */}
        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="px-3 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <span className="text-white font-semibold">{runs.length}</span> Sweeps Loaded
          </span>
          <span className="px-3 py-1 rounded-md bg-slate-900 border border-slate-800 text-emerald-300 flex items-center gap-1.5">
            <Award className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-white font-semibold">
              {paretoPoints.filter((p) => p.is_pareto_optimal).length}
            </span> Pareto Optimal
          </span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Monolingual Peak (EN)</span>
            <span className="p-1.5 rounded-lg bg-blue-500/10 text-blue-400">
              <TrendingUp className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-white">98.0%</div>
          <div className="mt-0.5 text-xs text-slate-500">Dense BGE-M3 · Latency: 38.4ms</div>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Monolingual Peak (BN)</span>
            <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400">
              <TrendingUp className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-white">92.0%</div>
          <div className="mt-0.5 text-xs text-slate-500">Dense BGE-M3 · Latency: 42.1ms</div>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">RQ3 Hypothesis Family</span>
            <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400">
              <Sparkles className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-indigo-300">H1 Supported</div>
          <div className="mt-0.5 text-xs text-slate-500">
            EN→BN penalty +0.8613 · BN→EN gain -0.8382 · H2–H4 Not Supported
          </div>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Fastest Top-5 Retrieval</span>
            <span className="p-1.5 rounded-lg bg-amber-500/10 text-amber-400">
              <Clock className="w-4 h-4" />
            </span>
          </div>
          <div className="mt-2 text-2xl font-bold font-mono text-amber-300">12.1 ms</div>
          <div className="mt-0.5 text-xs text-slate-500">BM25 Inverted Index · EN-EN</div>
        </div>
      </div>

      {/* Controlled Evaluation Invariant Banner */}
      <div className="p-3.5 bg-slate-900/90 rounded-xl border border-indigo-500/30 text-xs flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-slate-300">
          <span className="p-1 rounded bg-indigo-500/20 text-indigo-300 font-mono text-[10px] font-bold">
            CONTROLLED COMPARISON
          </span>
          <span>
            SciFact Multilingual (N=25 Matched Information Units) · Protocol: <span className="font-mono text-white">ragbench-protocol-v1.0</span> · Baseline: <span className="font-mono text-white">eb3bb4d</span>
          </span>
        </div>
        <div className="text-[11px] text-slate-400 font-mono">
          Strict scientific comparability: depth k=5, identical query distribution
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400 flex items-center gap-1 font-medium mr-1">
            <Filter className="w-3.5 h-3.5" /> Filters:
          </span>

          {/* Modality Chips */}
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
            {['ALL', 'EN-EN', 'BN-BN', 'EN-BN', 'BN-EN'].map((mod) => (
              <button
                key={mod}
                onClick={() => setSelectedModality(mod)}
                className={`px-2 py-0.5 rounded text-xs font-mono transition-colors ${
                  selectedModality === mod
                    ? 'bg-indigo-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {mod}
              </button>
            ))}
          </div>

          {/* Strategy Chips */}
          <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
            {['ALL', 'dense', 'bm25', 'hybrid'].map((strat) => (
              <button
                key={strat}
                onClick={() => setSelectedStrategy(strat)}
                className={`px-2 py-0.5 rounded text-xs font-mono transition-colors ${
                  selectedStrategy === strat
                    ? 'bg-indigo-600 text-white font-semibold'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                {strat.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="text-xs text-slate-500 font-mono">
          Showing {filteredRuns.length} of {runs.length} runs
        </div>
      </div>

      {/* Visualizations Grid: Pareto Frontier (Left) + Radar Chart (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6">
          <ParetoFrontierPlot
            points={filteredParetoPoints}
            selectedRunId={highlightedRunId}
            onSelectRun={(id) => setHighlightedRunId(id)}
          />
        </div>
        <div className="lg:col-span-6">
          <RadarChart
            data={radarData}
            activeRunIds={activeRadarIds}
            onToggleRun={toggleRadarRun}
          />
        </div>
      </div>

      {/* Comprehensive Sweep Table */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Full Experimental Run Matrix</h3>
          <span className="text-xs text-slate-400">
            Click table headers to sort · Toggle checkbox to add/remove from Radar Chart
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-950/70 border-b border-slate-800 text-slate-400 font-mono">
              <tr>
                <th className="py-2.5 px-3 w-10 text-center">Radar</th>
                <th
                  onClick={() => handleSort('name')}
                  className="py-2.5 px-3 cursor-pointer hover:text-white"
                >
                  Experiment / Configuration
                </th>
                <th
                  onClick={() => handleSort('modality')}
                  className="py-2.5 px-3 cursor-pointer hover:text-white"
                >
                  Modality
                </th>
                <th
                  onClick={() => handleSort('strategy')}
                  className="py-2.5 px-3 cursor-pointer hover:text-white"
                >
                  Topology
                </th>
                <th
                  onClick={() => handleSort('recall_at_5')}
                  className="py-2.5 px-3 cursor-pointer hover:text-white text-right"
                >
                  Recall@5 {sortField === 'recall_at_5' ? (sortAsc ? '↑' : '↓') : ''}
                </th>
                <th
                  onClick={() => handleSort('mrr_at_5')}
                  className="py-2.5 px-3 cursor-pointer hover:text-white text-right"
                >
                  MRR@5 {sortField === 'mrr_at_5' ? (sortAsc ? '↑' : '↓') : ''}
                </th>
                <th
                  onClick={() => handleSort('ndcg_at_5')}
                  className="py-2.5 px-3 cursor-pointer hover:text-white text-right"
                >
                  NDCG@5 {sortField === 'ndcg_at_5' ? (sortAsc ? '↑' : '↓') : ''}
                </th>
                <th
                  onClick={() => handleSort('latency_ms')}
                  className="py-2.5 px-3 cursor-pointer hover:text-white text-right"
                >
                  Latency {sortField === 'latency_ms' ? (sortAsc ? '↑' : '↓') : ''}
                </th>
                <th className="py-2.5 px-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
              {filteredRuns.map((r) => {
                const isRadarActive = activeRadarIds.includes(r.run_id)
                const isSelected = highlightedRunId === r.run_id
                const isPareto = paretoPoints.find((p) => p.run_id === r.run_id)?.is_pareto_optimal

                return (
                  <tr
                    key={r.run_id}
                    onClick={() => setHighlightedRunId(r.run_id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-indigo-950/40 text-white'
                        : 'hover:bg-slate-800/40 text-slate-300'
                    }`}
                  >
                    <td className="py-2.5 px-3 text-center" onClick={(e) => e.stopPropagation()}>
                      <input
                        type="checkbox"
                        checked={isRadarActive}
                        onChange={() => toggleRadarRun(r.run_id)}
                        className="rounded border-slate-700 text-indigo-600 focus:ring-0 cursor-pointer"
                      />
                    </td>
                    <td className="py-2.5 px-3 font-medium text-white">
                      {r.name}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${getModalityBadge(r.modality)}`}>
                        {r.modality}
                      </span>
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-semibold border ${getStrategyBadge(r.strategy)}`}>
                        {r.strategy}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-bold text-white">
                      {(r.recall_at_5 * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-300">
                      {r.mrr_at_5.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-300">
                      {r.ndcg_at_5.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-right text-slate-400">
                      {r.latency_ms.toFixed(1)} ms
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      {isPareto ? (
                        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px]">
                          <CheckCircle2 className="w-3 h-3" /> Pareto
                        </span>
                      ) : (
                        <span className="text-slate-500 text-[10px]">Dominated</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
