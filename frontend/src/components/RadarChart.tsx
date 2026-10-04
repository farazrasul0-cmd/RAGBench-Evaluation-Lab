import React, { useState } from 'react'
import type { RadarMetricData } from '../types'

interface RadarChartProps {
  data: RadarMetricData[]
  activeRunIds?: string[]
  onToggleRun?: (runId: string) => void
}

const AXIS_KEYS: (keyof RadarMetricData['metrics'])[] = [
  'recall_at_10',
  'precision_at_5',
  'faithfulness',
  'citation_accuracy',
  'cost_efficiency',
]

const AXIS_LABELS: Record<keyof RadarMetricData['metrics'], string> = {
  recall_at_10: 'Recall@10',
  precision_at_5: 'Precision@5',
  faithfulness: 'Faithfulness',
  citation_accuracy: 'Citation Acc.',
  cost_efficiency: 'Cost Efficiency',
}

export const RadarChart: React.FC<RadarChartProps> = ({
  data,
  activeRunIds,
  onToggleRun,
}) => {
  const [hoveredMetric, setHoveredMetric] = useState<{
    runName: string
    axis: string
    value: number
  } | null>(null)

  const size = 390
  const cx = size / 2
  const cy = size / 2
  const radius = 125
  const numAxes = AXIS_KEYS.length

  // Compute angle for axis i (-90 deg offset so 0 points upwards)
  const getAngle = (index: number) => {
    return (2 * Math.PI * index) / numAxes - Math.PI / 2
  }

  // Get (x, y) point for given radius and angle
  const getPoint = (r: number, angle: number) => {
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
    }
  }

  // Generate pentagon grid polygon points for given scale (0.2, 0.4, 0.6, 0.8, 1.0)
  const getGridPolygon = (scale: number) => {
    const points = []
    for (let i = 0; i < numAxes; i++) {
      const angle = getAngle(i)
      const pt = getPoint(radius * scale, angle)
      points.push(`${pt.x},${pt.y}`)
    }
    return points.join(' ')
  }

  // Filter visible series
  const visibleSeries = data.filter((item) =>
    activeRunIds ? activeRunIds.includes(item.run_id) : true
  )

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h3 className="text-sm font-semibold text-white">Multi-Dimensional Metric Radar</h3>
          <p className="text-xs text-slate-400">Comparing trade-offs across 5 core evaluation axes</p>
        </div>
        {hoveredMetric && (
          <div className="text-xs font-mono bg-slate-950 px-2 py-0.5 rounded border border-slate-800 text-indigo-300">
            {hoveredMetric.runName} · {hoveredMetric.axis}: {(hoveredMetric.value * 100).toFixed(1)}%
          </div>
        )}
      </div>

      <div className="w-full flex justify-center py-1">
        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="w-full max-w-[360px] h-auto select-none"
        >
          {/* Concentric Pentagon Grid Rings */}
          {[0.2, 0.4, 0.6, 0.8, 1.0].map((scale) => (
            <polygon
              key={`ring-${scale}`}
              points={getGridPolygon(scale)}
              fill="none"
              stroke="#334155"
              strokeWidth="0.8"
              strokeDasharray={scale === 1.0 ? 'none' : '3 3'}
            />
          ))}

          {/* Grid Scale Labels on Top Axis */}
          {[0.4, 0.8, 1.0].map((scale) => (
            <text
              key={`scale-${scale}`}
              x={cx + 4}
              y={cy - radius * scale - 2}
              className="text-[9px] fill-slate-500 font-mono"
            >
              {scale.toFixed(1)}
            </text>
          ))}

          {/* Axis Spoke Lines and Labels */}
          {AXIS_KEYS.map((key, i) => {
            const angle = getAngle(i)
            const outerPoint = getPoint(radius, angle)
            const labelPoint = getPoint(radius + 20, angle)

            // Text alignment based on horizontal position
            let textAnchor: 'middle' | 'start' | 'end' = 'middle'
            if (Math.cos(angle) > 0.3) textAnchor = 'start'
            else if (Math.cos(angle) < -0.3) textAnchor = 'end'

            return (
              <g key={`axis-${key}`}>
                <line
                  x1={cx}
                  y1={cy}
                  x2={outerPoint.x}
                  y2={outerPoint.y}
                  stroke="#475569"
                  strokeWidth="1"
                />
                <text
                  x={labelPoint.x}
                  y={labelPoint.y + 4}
                  textAnchor={textAnchor}
                  className="text-[10px] font-mono fill-slate-300 font-medium"
                >
                  {AXIS_LABELS[key]}
                </text>
              </g>
            )
          })}

          {/* Data Polygons */}
          {visibleSeries.map((series) => {
            const pointsString = AXIS_KEYS.map((key, i) => {
              const val = Math.max(0, Math.min(1, series.metrics[key]))
              const angle = getAngle(i)
              const pt = getPoint(radius * val, angle)
              return `${pt.x},${pt.y}`
            }).join(' ')

            return (
              <g key={`series-${series.run_id}`}>
                <polygon
                  points={pointsString}
                  fill={series.color}
                  fillOpacity="0.22"
                  stroke={series.color}
                  strokeWidth="2"
                  className="transition-all duration-200"
                />

                {/* Vertex Markers */}
                {AXIS_KEYS.map((key, i) => {
                  const val = Math.max(0, Math.min(1, series.metrics[key]))
                  const angle = getAngle(i)
                  const pt = getPoint(radius * val, angle)

                  return (
                    <circle
                      key={`series-${series.run_id}-pt-${key}`}
                      cx={pt.x}
                      cy={pt.y}
                      r="4"
                      fill={series.color}
                      stroke="#0f172a"
                      strokeWidth="1.5"
                      className="cursor-pointer hover:r-5 transition-all"
                      onMouseEnter={() =>
                        setHoveredMetric({
                          runName: series.name,
                          axis: AXIS_LABELS[key],
                          value: val,
                        })
                      }
                      onMouseLeave={() => setHoveredMetric(null)}
                    />
                  )
                })}
              </g>
            )
          })}
        </svg>
      </div>

      {/* Series Toggle Legend */}
      <div className="flex flex-wrap items-center justify-center gap-3 pt-3 border-t border-slate-800 text-xs">
        {data.map((series) => {
          const isVisible = activeRunIds ? activeRunIds.includes(series.run_id) : true
          return (
            <button
              key={series.run_id}
              onClick={() => onToggleRun?.(series.run_id)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-mono transition-all ${
                isVisible
                  ? 'bg-slate-800 text-white border border-slate-700'
                  : 'bg-slate-950 text-slate-500 border border-slate-900 opacity-60'
              }`}
            >
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: series.color }}
              />
              <span className="truncate max-w-[120px]">{series.name}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
