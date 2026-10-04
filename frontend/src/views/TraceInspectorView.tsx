import React, { useState, useEffect, useMemo, useRef } from 'react'
import {
  FileSearch,
  Search,
  CheckCircle2,
  AlertTriangle,
  ExternalLink,
  Clock,
  Sparkles,
  BookOpen,
} from 'lucide-react'
import type { QueryTraceDetail } from '../types'
import { fetchTraces } from '../api/client'

export const TraceInspectorView: React.FC = () => {
  const [traces, setTraces] = useState<QueryTraceDetail[]>([])
  const [selectedTraceId, setSelectedTraceId] = useState<string>('')
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [modalityFilter, setModalityFilter] = useState<string>('ALL')
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null)
  const [unresolvedCitation, setUnresolvedCitation] = useState<string | null>(null)
  
  const chunkRefs = useRef<Record<string, HTMLDivElement | null>>({})

  useEffect(() => {
    let isMounted = true
    
    fetchTraces()
      .then((data) => {
        if (isMounted) {
          setTraces(data)
          if (data.length > 0) {
            setSelectedTraceId(data[0].query_run_id)
          }
                  }
      })
      .catch(() => {})

    return () => {
      isMounted = false
    }
  }, [])

  // Filtered trace list
  const filteredTraces = useMemo(() => {
    return traces.filter((t) => {
      const matchMod = modalityFilter === 'ALL' || t.modality === modalityFilter
      const matchSearch =
        searchQuery === '' ||
        t.original_query.toLowerCase().includes(searchQuery.toLowerCase()) ||
        t.query_id.toLowerCase().includes(searchQuery.toLowerCase())
      return matchMod && matchSearch
    })
  }, [traces, modalityFilter, searchQuery])

  // Active selected trace
  const activeTrace = useMemo(() => {
    return traces.find((t) => t.query_run_id === selectedTraceId) || traces[0] || null
  }, [traces, selectedTraceId])

  const scrollToChunk = (chunkId: string) => {
    // Explicit provenance verification: citation must resolve to an actual retrieved chunk
    const chunkExists = activeTrace?.retrieved_chunks.some(
      (c) => c.chunk_id === chunkId || c.doc_id === chunkId
    )

    if (!chunkExists) {
      setUnresolvedCitation(chunkId)
      setHighlightedChunkId(null)
      return
    }

    setUnresolvedCitation(null)
    setHighlightedChunkId(chunkId)
    const el = chunkRefs.current[chunkId]
    if (el && typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }

  const getModalityColor = (mod: string) => {
    switch (mod) {
      case 'EN-EN':
        return 'text-blue-400 bg-blue-500/10 border-blue-500/30'
      case 'BN-BN':
        return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
      case 'EN-BN':
        return 'text-amber-400 bg-amber-500/10 border-amber-500/30'
      case 'BN-EN':
        return 'text-rose-400 bg-rose-500/10 border-rose-500/30'
      default:
        return 'text-slate-400 bg-slate-800 border-slate-700'
    }
  }

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <FileSearch className="w-5 h-5 text-indigo-400" />
            Per-Question Trace & Hallucination Inspector
          </h2>
          <p className="text-sm text-slate-400">
            Granular sentence-level verification, retrieval rank attribution, and claim-level hallucination audits.
          </p>
        </div>

        {/* Global summary badge */}
        <div className="flex items-center gap-2 font-mono text-xs text-slate-400">
          <span className="px-3 py-1 rounded bg-slate-900 border border-slate-800">
            Dataset: <span className="text-white font-semibold">SciFact Matched Units (N=25)</span>
          </span>
        </div>
      </div>

      {/* Main Grid: Query List Sidebar (4 Cols) + Deep Trace Detail (8 Cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Trace Selector Sidebar */}
        <div className="lg:col-span-4 space-y-3">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-3 space-y-3">
            {/* Search Input */}
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search queries or IDs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-1.5 text-xs bg-slate-950 border border-slate-800 rounded-lg text-white focus:outline-none focus:border-indigo-500"
              />
            </div>

            {/* Modality Filter Pills */}
            <div className="flex items-center gap-1 overflow-x-auto pb-1">
              {['ALL', 'EN-EN', 'BN-BN', 'EN-BN', 'BN-EN'].map((mod) => (
                <button
                  key={mod}
                  onClick={() => setModalityFilter(mod)}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono whitespace-nowrap transition-colors ${
                    modalityFilter === mod
                      ? 'bg-indigo-600 text-white font-semibold'
                      : 'bg-slate-950 text-slate-400 hover:text-white border border-slate-800/80'
                  }`}
                >
                  {mod}
                </button>
              ))}
            </div>
          </div>

          {/* Trace List */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden max-h-[620px] overflow-y-auto divide-y divide-slate-800/60">
            {filteredTraces.map((trace) => {
              const isSelected = activeTrace?.query_run_id === trace.query_run_id
              return (
                <div
                  key={trace.query_run_id}
                  onClick={() => setSelectedTraceId(trace.query_run_id)}
                  className={`p-3 cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-indigo-950/40 border-l-4 border-indigo-500 text-white'
                      : 'hover:bg-slate-800/40 text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] mb-1">
                    <span className="font-mono text-slate-400">{trace.query_id}</span>
                    <span className={`px-1.5 py-0.2 rounded font-mono font-semibold border text-[10px] ${getModalityColor(trace.modality)}`}>
                      {trace.modality}
                    </span>
                  </div>
                  <div className="text-xs font-medium text-slate-200 line-clamp-2 leading-snug">
                    {trace.original_query}
                  </div>
                  <div className="mt-2 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" /> {trace.latency_ms.toFixed(1)} ms
                    </span>
                    <span className="text-emerald-400">
                      R@5: {(trace.metric_scores.recall_at_5 * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              )
            })}
            {filteredTraces.length === 0 && (
              <div className="p-8 text-center text-slate-500 text-xs">
                No query traces matching your filter.
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Deep Trace Inspection Details */}
        <div className="lg:col-span-8 space-y-5">
          {activeTrace ? (
            <>
              {/* Header Box: Active Query Details */}
              <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-slate-400">{activeTrace.query_id}</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-mono border font-semibold ${getModalityColor(activeTrace.modality)}`}>
                      {activeTrace.modality}
                    </span>
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                      {activeTrace.domain}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-slate-400 font-mono">
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                    <span>{activeTrace.latency_ms.toFixed(1)} ms</span>
                  </div>
                </div>

                <div>
                  <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                    Evaluated Query
                  </div>
                  <div className="text-sm font-medium text-white bg-slate-950 p-2.5 rounded-lg border border-slate-800/80 leading-relaxed font-sans">
                    {activeTrace.original_query}
                  </div>
                </div>

                <div>
                  <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
                    Expected Target / Ground Truth Fact
                  </div>
                  <div className="text-xs text-slate-300 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/60 leading-relaxed">
                    {activeTrace.expected_answer}
                  </div>
                </div>
              </div>

              {/* Metric Scorecards */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 font-mono uppercase">Recall@5</div>
                  <div className="text-base font-bold font-mono text-emerald-400 mt-0.5">
                    {(activeTrace.metric_scores.recall_at_5 * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 font-mono uppercase">MRR@5</div>
                  <div className="text-base font-bold font-mono text-blue-400 mt-0.5">
                    {activeTrace.metric_scores.mrr_at_5.toFixed(2)}
                  </div>
                </div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 font-mono uppercase">NDCG@5</div>
                  <div className="text-base font-bold font-mono text-purple-400 mt-0.5">
                    {activeTrace.metric_scores.ndcg_at_5.toFixed(2)}
                  </div>
                </div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 font-mono uppercase">Precision@5</div>
                  <div className="text-base font-bold font-mono text-indigo-400 mt-0.5">
                    {(activeTrace.metric_scores.precision_at_5 * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 font-mono uppercase">Faithfulness</div>
                  <div className="text-base font-bold font-mono text-emerald-400 mt-0.5">
                    {(activeTrace.metric_scores.faithfulness * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 font-mono uppercase">Citation Prec.</div>
                  <div className="text-base font-bold font-mono text-teal-400 mt-0.5">
                    {(activeTrace.metric_scores.citation_precision * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              {/* Split Analysis: Generated Answer with Spans (Top) + Retrieved Chunks (Bottom) */}
              <div className="space-y-4">
                {/* Generation with Sentence / Span Highlighting */}
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                      Generated Answer & Claim-Level Attribution
                    </div>
                    <div className="flex items-center gap-3 text-[11px] font-mono">
                      <span className="flex items-center gap-1 text-emerald-400">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Supported Claim
                      </span>
                      <span className="flex items-center gap-1 text-rose-400">
                        <AlertTriangle className="w-3.5 h-3.5" /> Unsupported / Hallucinated
                      </span>
                    </div>
                  </div>

                  <div className="p-3.5 bg-slate-950 rounded-lg border border-slate-800 text-xs leading-relaxed space-y-2 font-sans">
                    {/* Unresolved Citation Provenance Alert Banner */}
                    {unresolvedCitation && (
                      <div className="p-3 bg-rose-950/40 border border-rose-500/50 rounded-lg text-rose-200 text-xs flex items-center justify-between gap-3 mb-2">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                          <div>
                            <span className="font-semibold font-mono">Unresolved provenance:</span>
                            <span className="ml-1">
                              Citation <code className="bg-rose-900/50 px-1 py-0.2 rounded text-white font-mono font-bold">[{unresolvedCitation}]</code> could not be matched to a retrieved passage in this trace.
                            </span>
                          </div>
                        </div>
                        <button
                          onClick={() => setUnresolvedCitation(null)}
                          className="text-[11px] text-rose-300 hover:text-white px-2 py-0.5 rounded bg-rose-900/40 hover:bg-rose-800 transition-colors"
                        >
                          Dismiss
                        </button>
                      </div>
                    )}

                    {activeTrace.annotated_spans.map((span, idx) => {
                      const isEntailing = span.is_entailing
                      return (
                        <div
                          key={`span-${idx}`}
                          className={`p-2 rounded border transition-all ${
                            isEntailing
                              ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-100'
                              : 'bg-rose-950/20 border-rose-500/40 text-rose-100'
                          }`}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <span className="flex-1">{span.text}</span>
                            <div className="flex items-center gap-1 shrink-0 font-mono">
                              {span.citation_ids.map((cit) => {
                                const isResolvable = activeTrace.retrieved_chunks.some(
                                  (c) => c.chunk_id === cit || c.doc_id === cit
                                )
                                return (
                                  <button
                                    key={cit}
                                    onClick={() => scrollToChunk(cit)}
                                    className={`px-1.5 py-0.5 rounded text-[10px] border transition-colors flex items-center gap-0.5 ${
                                      isResolvable
                                        ? 'bg-slate-800 hover:bg-indigo-600 text-indigo-300 hover:text-white border-slate-700'
                                        : 'bg-amber-950/40 hover:bg-rose-900 text-amber-300 hover:text-white border-amber-500/40'
                                    }`}
                                    title={isResolvable ? `Jump to chunk ${cit}` : `Unresolved provenance: ${cit}`}
                                  >
                                    <span>[{cit.split('_').slice(-2).join('_')}]</span>
                                    {isResolvable ? <ExternalLink className="w-2.5 h-2.5" /> : <AlertTriangle className="w-2.5 h-2.5 text-amber-400" />}
                                  </button>
                                )
                              })}
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Retrieved Context Chunks */}
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                      <BookOpen className="w-3.5 h-3.5 text-blue-400" />
                      Retrieved Passage Chunks (Top-{activeTrace.retrieved_chunks.length})
                    </div>
                    <span className="text-[11px] text-slate-500 font-mono">
                      Ranked by similarity / RRF score
                    </span>
                  </div>

                  <div className="space-y-2.5 max-h-[440px] overflow-y-auto pr-1">
                    {activeTrace.retrieved_chunks.map((chunk) => {
                      const isHighlighted = highlightedChunkId === chunk.chunk_id
                      const isGroundTruth = activeTrace.ground_truth_chunks.includes(chunk.chunk_id)

                      return (
                        <div
                          key={chunk.chunk_id}
                          ref={(el) => {
                            chunkRefs.current[chunk.chunk_id] = el
                          }}
                          className={`p-3 rounded-lg border transition-all ${
                            isHighlighted
                              ? 'bg-indigo-950/60 border-indigo-400 ring-2 ring-indigo-500/50'
                              : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                          }`}
                        >
                          <div className="flex items-center justify-between text-xs mb-1.5">
                            <div className="flex items-center gap-2">
                              <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px] font-bold">
                                #{chunk.rank}
                              </span>
                              <span className="font-mono text-slate-400 text-[11px]">{chunk.chunk_id}</span>
                              {isGroundTruth && (
                                <span className="px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 text-[10px] font-mono">
                                  Ground Truth
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className="bg-indigo-500 h-1.5 rounded-full"
                                  style={{ width: `${Math.min(100, chunk.score * 100)}%` }}
                                />
                              </div>
                              <span className="font-mono text-[11px] text-slate-300 font-bold">
                                {chunk.score.toFixed(3)}
                              </span>
                            </div>
                          </div>
                          <div className="text-xs text-slate-300 font-serif leading-relaxed pl-1 border-l-2 border-slate-800">
                            {chunk.content}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="p-12 text-center text-slate-500 bg-slate-900 rounded-xl border border-slate-800">
              No query trace selected.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
