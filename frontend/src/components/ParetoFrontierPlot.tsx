import React, { useState, useMemo } from 'react'
import type { ParetoPoint } from '../types'
import { validateControlledComparison } from '../api/client'

interface ParetoFrontierPlotProps {
  points: ParetoPoint[]
  selectedRunId?: string
  onSelectRun?: (runId: string) => void
}

export const ParetoFrontierPlot: React.FC<ParetoFrontierPlotProps> = ({
  points,
  selectedRunId,
  onSelectRun,
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<ParetoPoint | null>(null)

  // Chart dimensions & margins
  const width = 580
  const height = 360
  const margin = { top: 30, right: 30, bottom: 50, left: 60 }
  const innerWidth = width - margin.left - margin.right
  const innerHeight = height - margin.top - margin.bottom

  // Min / max calculations with padding
  const { minLatency, maxLatency, minRecall, maxRecall } = useMemo(() => {
    if (points.length === 0) {
      return { minLatency: 0, maxLatency: 100, minRecall: 0, maxRecall: 1 }
    }
    const lats = points.map((p) => p.latency_ms)
    const recs = points.map((p) => p.recall_at_5)
    const minL = Math.max(0, Math.floor(Math.min(...lats) * 0.8))
    const maxL = Math.ceil(Math.max(...lats) * 1.15)
    const minR = Math.max(0, Math.floor(Math.min(...recs) * 10 - 1) / 10)
    const maxR = 1.05
    return { minLatency: minL, maxLatency: maxL, minRecall: minR, maxRecall: maxR }
  }, [points])

  // Validate controlled comparison conditions across points
  const controlValidation = useMemo(() => {
    return validateControlledComparison(points)
  }, [points])

  // Calculate and sort Pareto optimal points (only valid when conditions are controlled)
  const paretoPoints = useMemo(() => {
    if (!controlValidation.is_controlled) return []
    return points
      .filter((p) => p.is_pareto_optimal)
      .sort((a, b) => a.latency_ms - b.latency_ms)
  }, [controlValidation.is_controlled, points])

  // Coordinate scales
  const xScale = (lat: number) => {
    return margin.left + ((lat - minLatency) / (maxLatency - minLatency || 1)) * innerWidth
  }

  const yScale = (rec: number) => {
    return margin.top + innerHeight - ((rec - minRecall) / (maxRecall - minRecall || 1)) * innerHeight
  }

  // Build SVG path for frontier curve (suppressed if conditions are uncontrolled)
  const frontierPath = useMemo(() => {
    if (!controlValidation.is_controlled || paretoPoints.length === 0) return ''
    return paretoPoints.reduce((acc, pt, idx) => {
      const x = margin.left + ((pt.latency_ms - minLatency) / (maxLatency - minLatency || 1)) * innerWidth
      const y = margin.top + innerHeight - ((pt.recall_at_5 - minRecall) / (maxRecall - minRecall || 1)) * innerHeight
      return idx === 0 ? `M ${x},${y}` : `${acc} L ${x},${y}`
    }, '')
  }, [controlValidation.is_controlled, paretoPoints, minLatency, maxLatency, minRecall, maxRecall, margin.left, margin.top, innerWidth, innerHeight])

  const getColor = (strategy: string) => {
    switch (strategy.toLowerCase()) {
      case 'dense':
        return '#3b82f6' // blue-500
      case 'bm25':
        return '#f59e0b' // amber-500
      case 'hybrid':
        return '#a855f7' // purple-500
      default:
        return '#94a3b8' // slate-400
    }
  }

  // Ticks
  const xTicks = [minLatency, Math.round((minLatency + maxLatency) / 2), maxLatency]
  const yTicks = [0.4, 0.6, 0.8, 1.0].filter((v) => v >= minRecall && v <= maxRecall)

  return (
    <div className="relative bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white">Pareto Efficiency Frontier</h3>
            {controlValidation.is_controlled ? (
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 font-mono">
                Controlled Validated
              </span>
            ) : (
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30 font-mono">
                Incomparable Conditions
              </span>
            )}
          </div>
          <p className="text-xs text-slate-400">Recall@5 vs. End-to-End Latency (ms)</p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
            <span className="text-slate-300">Dense</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
            <span className="text-slate-300">BM25</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
            <span className="text-slate-300">Hybrid</span>
          </div>
          <div className="flex items-center gap-1.5 border-l border-slate-700 pl-2">
            <span className="w-3 h-0.5 bg-emerald-400 border border-emerald-400" />
            <span className="text-emerald-400 font-mono text-[11px]">Frontier</span>
          </div>
        </div>
      </div>

      {!controlValidation.is_controlled && (
        <div className="mb-2 p-2.5 rounded-lg bg-rose-950/40 border border-rose-500/40 text-[11px] text-rose-200 flex items-center gap-2">
          <span>🛑 Incomparable Evaluation Conditions: Selected sweeps differ in [{controlValidation.divergent_fields.join(', ')}]. Frontier calculation is strictly disabled to prevent invalid scientific comparisons.</span>
        </div>
      )}

      <div className="w-full overflow-hidden flex justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full max-w-full h-auto select-none"
          style={{ minHeight: '300px' }}
        >
          {/* Grid lines */}
          {xTicks.map((xVal) => (
            <line
              key={`x-grid-${xVal}`}
              x1={xScale(xVal)}
              y1={margin.top}
              x2={xScale(xVal)}
              y2={margin.top + innerHeight}
              stroke="#334155"
              strokeDasharray="3 3"
              strokeWidth="0.8"
            />
          ))}

          {yTicks.map((yVal) => (
            <line
              key={`y-grid-${yVal}`}
              x1={margin.left}
              y1={yScale(yVal)}
              x2={margin.left + innerWidth}
              y2={yScale(yVal)}
              stroke="#334155"
              strokeDasharray="3 3"
              strokeWidth="0.8"
            />
          ))}

          {/* Axes */}
          <line
            x1={margin.left}
            y1={margin.top + innerHeight}
            x2={margin.left + innerWidth}
            y2={margin.top + innerHeight}
            stroke="#64748b"
            strokeWidth="1.5"
          />
          <line
            x1={margin.left}
            y1={margin.top}
            x2={margin.left}
            y2={margin.top + innerHeight}
            stroke="#64748b"
            strokeWidth="1.5"
          />

          {/* X Tick Labels */}
          {xTicks.map((xVal) => (
            <text
              key={`x-label-${xVal}`}
              x={xScale(xVal)}
              y={margin.top + innerHeight + 18}
              textAnchor="middle"
              className="text-[10px] fill-slate-400 font-mono"
            >
              {xVal} ms
            </text>
          ))}

          {/* Y Tick Labels */}
          {yTicks.map((yVal) => (
            <text
              key={`y-label-${yVal}`}
              x={margin.left - 10}
              y={yScale(yVal) + 3}
              textAnchor="end"
              className="text-[10px] fill-slate-400 font-mono"
            >
              {(yVal * 100).toFixed(0)}%
            </text>
          ))}

          {/* Axis Titles */}
          <text
            x={margin.left + innerWidth / 2}
            y={height - 8}
            textAnchor="middle"
            className="text-[11px] fill-slate-400 font-sans"
          >
            Latency (ms, lower is better)
          </text>
          <text
            x={-height / 2}
            y={18}
            transform="rotate(-90)"
            textAnchor="middle"
            className="text-[11px] fill-slate-400 font-sans"
          >
            Retrieval Recall@5 (higher is better)
          </text>

          {/* Pareto Frontier Line */}
          {frontierPath && (
            <path
              d={frontierPath}
              fill="none"
              stroke="#10b981"
              strokeWidth="2.5"
              strokeLinejoin="round"
              className="drop-shadow-[0_0_8px_rgba(16,185,129,0.5)]"
            />
          )}

          {/* Data Points */}
          {points.map((pt) => {
            const cx = xScale(pt.latency_ms)
            const cy = yScale(pt.recall_at_5)
            const isHovered = hoveredPoint?.run_id === pt.run_id
            const isSelected = selectedRunId === pt.run_id
            const color = getColor(pt.strategy)

            return (
              <g
                key={pt.run_id}
                className="cursor-pointer transition-all"
                onMouseEnter={() => setHoveredPoint(pt)}
                onMouseLeave={() => setHoveredPoint(null)}
                onClick={() => onSelectRun?.(pt.run_id)}
              >
                {/* Highlight ring for Pareto optimal or selected */}
                {(pt.is_pareto_optimal || isSelected) && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSelected ? 10 : 8}
                    fill="none"
                    stroke={pt.is_pareto_optimal ? '#10b981' : color}
                    strokeWidth={isSelected ? '2.5' : '1.5'}
                    strokeDasharray={pt.is_pareto_optimal ? 'none' : '2 2'}
                    className="opacity-70 animate-pulse"
                  />
                )}

                {/* Point circle */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 7 : 5}
                  fill={color}
                  stroke="#0f172a"
                  strokeWidth="2"
                  className="transition-transform duration-150"
                />

                {/* Modality small tag */}
                {pt.modality && (
                  <text
                    x={cx + 7}
                    y={cy - 4}
                    className="text-[9px] fill-slate-300 font-mono font-bold pointer-events-none drop-shadow-md"
                  >
                    {pt.modality}
                  </text>
                )}
              </g>
            )
          })}
        </svg>
      </div>

      {/* Floating Hover Card */}
      {hoveredPoint && (
        <div className="absolute bottom-4 right-4 bg-slate-950/95 border border-slate-700 p-3 rounded-lg shadow-xl text-xs space-y-1 pointer-events-none z-10 backdrop-blur-sm max-w-xs">
          <div className="font-semibold text-white flex items-center justify-between gap-2">
            <span>{hoveredPoint.name}</span>
            {hoveredPoint.is_pareto_optimal && (
              <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.2 rounded font-mono">
                Optimal
              </span>
            )}
          </div>
          <div className="text-slate-400 font-mono text-[11px] grid grid-cols-2 gap-x-3 gap-y-0.5 pt-1 border-t border-slate-800">
            <div>Recall@5: <span className="text-white font-bold">{(hoveredPoint.recall_at_5 * 100).toFixed(1)}%</span></div>
            <div>Latency: <span className="text-white font-bold">{hoveredPoint.latency_ms.toFixed(1)} ms</span></div>
            <div>NDCG@5: <span className="text-white font-bold">{hoveredPoint.ndcg_at_5.toFixed(4)}</span></div>
            <div>Cost: <span className="text-white font-bold">${hoveredPoint.estimated_cost_usd.toFixed(4)}</span></div>
          </div>
        </div>
      )}
    </div>
  )
}
