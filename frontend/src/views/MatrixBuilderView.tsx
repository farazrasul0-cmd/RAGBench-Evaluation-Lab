import React, { useState, useMemo, useEffect } from 'react'
import {
  Layers,
  Cpu,
  Database,
  Sliders,
  Copy,
  Download,
  Check,
  Sparkles,
  FileCode,
  Table as TableIcon
} from 'lucide-react'
import type { MatrixSweepSelection, MatrixPreviewData } from '../types'
import { previewMatrix, exportMatrixYaml } from '../api/client'

const CHUNKING_OPTIONS = ['fixed', 'sentence', 'recursive', 'semantic']
const CHUNK_SIZE_OPTIONS = [200, 256, 512, 1024]
const OVERLAP_OPTIONS = [20, 32, 64, 128]
const EMBEDDING_OPTIONS = [
  'BAAI/bge-m3',
  'sentence-transformers/all-MiniLM-L6-v2',
  'text-embedding-3-small'
]
const RETRIEVAL_OPTIONS = ['dense', 'bm25', 'hybrid']
const RERANKER_OPTIONS = ['none', 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'bge-reranker-large']
const GENERATION_OPTIONS = ['none', 'gpt-4o-mini', 'claude-3-5-sonnet', 'gemini-1.5-pro']
const TOP_K_OPTIONS = [3, 5, 10]

export const MatrixBuilderView: React.FC = () => {
  const [name, setName] = useState('rq3_multilingual_factorial_sweep')
  const [description, setDescription] = useState('Factorial sweep evaluating cross-lingual transfer penalties across retrieval topologies.')
  const [datasetId, setDatasetId] = useState('scifact_multilingual')
  const [datasetVersionId, setDatasetVersionId] = useState('v1.0')

  const [chunking, setChunking] = useState<string[]>(['fixed_size', 'sentence'])
  const [chunkSizes, setChunkSizes] = useState<number[]>([512])
  const [chunkOverlaps, setChunkOverlaps] = useState<number[]>([64])
  const [embeddings, setEmbeddings] = useState<string[]>(['BAAI/bge-m3'])
  const [retrievals, setRetrievals] = useState<string[]>(['dense', 'bm25', 'hybrid'])
  const [rerankers, setRerankers] = useState<string[]>(['none'])
  const [generations, setGenerations] = useState<string[]>(['gpt-4o-mini'])
  const [topKValues, setTopKValues] = useState<number[]>([5])

  const [activeTab, setActiveTab] = useState<'table' | 'yaml'>('table')
  const [previewData, setPreviewData] = useState<MatrixPreviewData | null>(null)
  const [yamlContent, setYamlContent] = useState<string>('')
  const [copied, setCopied] = useState(false)
  
  const toggleItem = <T,>(list: T[], item: T, setter: React.Dispatch<React.SetStateAction<T[]>>) => {
    if (list.includes(item)) {
      if (list.length > 1) {
        setter(list.filter((x) => x !== item))
      }
    } else {
      setter([...list, item])
    }
  }

  const selection: MatrixSweepSelection = useMemo(() => ({
    name,
    description,
    dataset_id: datasetId,
    dataset_version_id: datasetVersionId,
    chunking_strategies: chunking,
    chunk_sizes: chunkSizes,
    chunk_overlaps: chunkOverlaps,
    embedding_models: embeddings,
    retrieval_strategies: retrievals,
    rerankers,
    generation_models: generations,
    top_k_values: topKValues,
  }), [name, description, datasetId, datasetVersionId, chunking, chunkSizes, chunkOverlaps, embeddings, retrievals, rerankers, generations, topKValues])

  const localTotalCombinations = useMemo(() => {
    return (
      chunking.length *
      chunkSizes.length *
      chunkOverlaps.length *
      embeddings.length *
      retrievals.length *
      rerankers.length *
      generations.length *
      topKValues.length
    )
  }, [chunking, chunkSizes, chunkOverlaps, embeddings, retrievals, rerankers, generations, topKValues])

  useEffect(() => {
    let isMounted = true
    
    Promise.all([
      previewMatrix(selection),
      exportMatrixYaml(selection)
    ]).then(([preview, yaml]) => {
      if (isMounted) {
        setPreviewData(preview)
        setYamlContent(typeof yaml === 'string' ? yaml : (yaml as { yaml_string?: string })?.yaml_string ?? '')
              }
    }).catch(() => {})

    return () => {
      isMounted = false
    }
  }, [selection])

  const handleCopyYaml = () => {
    navigator.clipboard.writeText(yamlContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownloadYaml = () => {
    const blob = new Blob([yamlContent], { type: 'text/yaml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${name || 'matrix_config'}.yaml`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  const loadPreset = (type: 'rq3' | 'fast' | 'full') => {
    if (type === 'rq3') {
      setName('rq3_multilingual_study_matrix')
      setDescription('Exact replication of approved RQ3 BAAI/bge-m3 12-run factorial sweep (eb3bb4d) evaluating cross-lingual transfer on N=25 matched units (fixed 200/20).')
      setDatasetId('multilingual_canonical')
      setDatasetVersionId('dd59f087-86ff-4872-925f-adb03fc8d9a2')
      setChunking(['fixed'])
      setChunkSizes([200])
      setChunkOverlaps([20])
      setEmbeddings(['BAAI/bge-m3'])
      setRetrievals(['dense', 'bm25', 'hybrid'])
      setRerankers(['none'])
      setGenerations(['none'])
      setTopKValues([5])
    } else if (type === 'fast') {
      setName('quick_smoke_matrix')
      setChunking(['fixed_size'])
      setChunkSizes([512])
      setChunkOverlaps([32])
      setEmbeddings(['sentence-transformers/all-MiniLM-L6-v2'])
      setRetrievals(['dense'])
      setRerankers(['none'])
      setGenerations(['gpt-4o-mini'])
      setTopKValues([5])
    } else {
      setName('exhaustive_factorial_matrix')
      setChunking(['fixed_size', 'sentence', 'recursive'])
      setChunkSizes([256, 512, 1024])
      setChunkOverlaps([32, 64])
      setEmbeddings(['BAAI/bge-m3', 'sentence-transformers/all-MiniLM-L6-v2'])
      setRetrievals(['dense', 'bm25', 'hybrid'])
      setRerankers(['none', 'cross-encoder/ms-marco-MiniLM-L-6-v2'])
      setGenerations(['gpt-4o-mini', 'claude-3-5-sonnet'])
      setTopKValues([5, 10])
    }
  }

  const complexityColor = useMemo(() => {
    if (localTotalCombinations < 20) return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
    if (localTotalCombinations <= 100) return 'text-amber-400 bg-amber-500/10 border-amber-500/30'
    return 'text-rose-400 bg-rose-500/10 border-rose-500/30'
  }, [localTotalCombinations])

  const complexityLabel = useMemo(() => {
    if (localTotalCombinations < 20) return 'LOW COMPLEXITY'
    if (localTotalCombinations <= 100) return 'MODERATE COMPLEXITY'
    return 'HEAVY COMBINATORIAL SWEEP'
  }, [localTotalCombinations])

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-400" />
            Experiment Matrix Builder
          </h2>
          <p className="text-sm text-slate-400">
            Define multi-dimensional parameter sweeps across chunking, embeddings, retrieval topologies, and rerankers.
          </p>
        </div>

        {/* Preset Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400 flex items-center gap-1 font-medium">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" /> Presets:
          </span>
          <button
            onClick={() => loadPreset('rq3')}
            className="px-2.5 py-1 text-xs font-medium rounded bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 transition-colors"
          >
            RQ3 Multilingual
          </button>
          <button
            onClick={() => loadPreset('fast')}
            className="px-2.5 py-1 text-xs font-medium rounded bg-slate-800 hover:bg-slate-700 text-emerald-300 border border-slate-700 transition-colors"
          >
            Fast Smoke
          </button>
          <button
            onClick={() => loadPreset('full')}
            className="px-2.5 py-1 text-xs font-medium rounded bg-slate-800 hover:bg-slate-700 text-purple-300 border border-slate-700 transition-colors"
          >
            Full Factorial
          </button>
        </div>
      </div>

      {/* Main Grid: Parameters on Left, Output & Preview on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Parameter Form (5 Cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Metadata Box */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Experiment Metadata</div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">Experiment Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-1.5 text-sm bg-slate-950 border border-slate-700 rounded-lg text-white font-mono focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="block text-xs text-slate-300 mb-1">Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="w-full px-3 py-1.5 text-xs bg-slate-950 border border-slate-700 rounded-lg text-slate-300 focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-slate-300 mb-1">Dataset ID</label>
                <input
                  type="text"
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                  className="w-full px-2.5 py-1 text-xs bg-slate-950 border border-slate-700 rounded-lg text-white font-mono"
                />
              </div>
              <div>
                <label className="block text-xs text-slate-300 mb-1">Dataset Version</label>
                <input
                  type="text"
                  value={datasetVersionId}
                  onChange={(e) => setDatasetVersionId(e.target.value)}
                  className="w-full px-2.5 py-1 text-xs bg-slate-950 border border-slate-700 rounded-lg text-white font-mono"
                />
              </div>
            </div>
          </div>

          {/* Dimension Selectors */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center justify-between">
              <span>Factorial Dimensions</span>
              <span className="text-indigo-400 font-mono text-[11px]">{localTotalCombinations} Variations</span>
            </div>

            {/* Chunking Strategies */}
            <div>
              <div className="text-xs text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-blue-400" /> Chunking Strategy ({chunking.length})
              </div>
              <div className="flex flex-wrap gap-1.5">
                {CHUNKING_OPTIONS.map((opt) => {
                  const active = chunking.includes(opt)
                  return (
                    <button
                      key={opt}
                      onClick={() => toggleItem(chunking, opt, setChunking)}
                      className={`px-2.5 py-1 rounded text-xs font-mono transition-all ${
                        active
                          ? 'bg-blue-600/30 text-blue-200 border border-blue-500/50 shadow-sm'
                          : 'bg-slate-950 text-slate-400 border border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      {opt}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Chunk Size & Overlap */}
            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-800/80">
              <div>
                <div className="text-xs text-slate-300 font-medium mb-1.5">Chunk Size (chars)</div>
                <div className="flex flex-wrap gap-1.5">
                  {CHUNK_SIZE_OPTIONS.map((opt) => {
                    const active = chunkSizes.includes(opt)
                    return (
                      <button
                        key={opt}
                        onClick={() => toggleItem(chunkSizes, opt, setChunkSizes)}
                        className={`px-2 py-0.5 rounded text-xs font-mono transition-all ${
                          active
                            ? 'bg-blue-600/30 text-blue-200 border border-blue-500/50'
                            : 'bg-slate-950 text-slate-400 border border-slate-800'
                        }`}
                      >
                        {opt}
                      </button>
                    )
                  })}
                </div>
              </div>

              <div>
                <div className="text-xs text-slate-300 font-medium mb-1.5">Overlap</div>
                <div className="flex flex-wrap gap-1.5">
                  {OVERLAP_OPTIONS.map((opt) => {
                    const active = chunkOverlaps.includes(opt)
                    return (
                      <button
                        key={opt}
                        onClick={() => toggleItem(chunkOverlaps, opt, setChunkOverlaps)}
                        className={`px-2 py-0.5 rounded text-xs font-mono transition-all ${
                          active
                            ? 'bg-blue-600/30 text-blue-200 border border-blue-500/50'
                            : 'bg-slate-950 text-slate-400 border border-slate-800'
                        }`}
                      >
                        {opt}
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Embedding Models */}
            <div className="pt-2 border-t border-slate-800/80">
              <div className="text-xs text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-emerald-400" /> Embedding Models ({embeddings.length})
              </div>
              <div className="flex flex-col gap-1.5">
                {EMBEDDING_OPTIONS.map((opt) => {
                  const active = embeddings.includes(opt)
                  return (
                    <button
                      key={opt}
                      onClick={() => toggleItem(embeddings, opt, setEmbeddings)}
                      className={`px-2.5 py-1.5 rounded text-xs font-mono text-left transition-all ${
                        active
                          ? 'bg-emerald-600/25 text-emerald-200 border border-emerald-500/50'
                          : 'bg-slate-950 text-slate-400 border border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      {opt}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Retrieval Topologies */}
            <div className="pt-2 border-t border-slate-800/80">
              <div className="text-xs text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-purple-400" /> Retrieval Strategies ({retrievals.length})
              </div>
              <div className="flex flex-wrap gap-1.5">
                {RETRIEVAL_OPTIONS.map((opt) => {
                  const active = retrievals.includes(opt)
                  return (
                    <button
                      key={opt}
                      onClick={() => toggleItem(retrievals, opt, setRetrievals)}
                      className={`px-2.5 py-1 rounded text-xs font-mono transition-all ${
                        active
                          ? 'bg-purple-600/30 text-purple-200 border border-purple-500/50'
                          : 'bg-slate-950 text-slate-400 border border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      {opt}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Reranker & Top-K */}
            <div className="pt-2 border-t border-slate-800/80">
              <div className="text-xs text-slate-300 font-medium mb-1.5 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-amber-400" /> Neural Rerankers ({rerankers.length})
              </div>
              <div className="flex flex-col gap-1.5">
                {RERANKER_OPTIONS.map((opt) => {
                  const active = rerankers.includes(opt)
                  return (
                    <button
                      key={opt}
                      onClick={() => toggleItem(rerankers, opt, setRerankers)}
                      className={`px-2.5 py-1.5 rounded text-xs font-mono text-left transition-all ${
                        active
                          ? 'bg-amber-600/25 text-amber-200 border border-amber-500/50'
                          : 'bg-slate-950 text-slate-400 border border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      {opt}
                    </button>
                  )
                })}
              </div>
            </div>

            {/* Generation Models & Top-K */}
            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-800/80">
              <div>
                <div className="text-xs text-slate-300 font-medium mb-1.5">Top-k Depth</div>
                <div className="flex flex-wrap gap-1.5">
                  {TOP_K_OPTIONS.map((opt) => {
                    const active = topKValues.includes(opt)
                    return (
                      <button
                        key={opt}
                        onClick={() => toggleItem(topKValues, opt, setTopKValues)}
                        className={`px-2 py-0.5 rounded text-xs font-mono transition-all ${
                          active
                            ? 'bg-indigo-600/30 text-indigo-200 border border-indigo-500/50'
                            : 'bg-slate-950 text-slate-400 border border-slate-800'
                        }`}
                      >
                        k={opt}
                      </button>
                    )
                  })}
                </div>
              </div>

              <div>
                <div className="text-xs text-slate-300 font-medium mb-1.5">LLM Generator</div>
                <div className="flex flex-col gap-1">
                  {GENERATION_OPTIONS.map((opt) => {
                    const active = generations.includes(opt)
                    return (
                      <button
                        key={opt}
                        onClick={() => toggleItem(generations, opt, setGenerations)}
                        className={`px-2 py-0.5 rounded text-[11px] font-mono text-left transition-all ${
                          active
                            ? 'bg-rose-600/25 text-rose-200 border border-rose-500/50'
                            : 'bg-slate-950 text-slate-400 border border-slate-800'
                        }`}
                      >
                        {opt}
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Combinatorial Summary + Output Tabs (7 Cols) */}
        <div className="lg:col-span-7 space-y-5">
          {/* Combinatorial Calculator Card */}
          <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Combinatorial Scale</div>
              <div className="flex items-center gap-2">
                {previewData?.compatibility_status && (
                  <span className={`text-[11px] px-2 py-0.5 rounded font-mono border font-semibold ${
                    previewData.compatibility_status === 'COMPATIBLE'
                      ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30'
                      : previewData.compatibility_status === 'WARNINGS'
                      ? 'text-amber-400 bg-amber-500/10 border-amber-500/30'
                      : 'text-rose-400 bg-rose-500/10 border-rose-500/30'
                  }`}>
                    {previewData.compatibility_status}
                  </span>
                )}
                <span className={`text-[11px] px-2.5 py-0.5 rounded-full font-mono border font-semibold ${complexityColor}`}>
                  {complexityLabel}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800/80">
                <div className="text-2xl font-bold font-mono text-indigo-300">{localTotalCombinations}</div>
                <div className="text-xs text-slate-400 mt-0.5">Factorial Pipelines</div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800/80">
                <div className="text-2xl font-bold font-mono text-emerald-300">
                  {localTotalCombinations * 25}
                </div>
                <div className="text-xs text-slate-400 mt-0.5">Est. Total Queries (N=25)</div>
              </div>
              <div className="p-3 bg-slate-950 rounded-lg border border-slate-800/80">
                <div className="text-2xl font-bold font-mono text-slate-300">
                  ~{(localTotalCombinations * 0.4).toFixed(1)}s
                </div>
                <div className="text-xs text-slate-400 mt-0.5">Est. Sweep Wallclock</div>
              </div>
            </div>
          </div>

          {/* Authoritative Backend Compatibility Diagnostics */}
          {previewData?.validation_warnings && previewData.validation_warnings.length > 0 && (
            <div className={`p-3.5 rounded-xl border text-xs space-y-1.5 ${
              previewData.is_executable
                ? 'bg-amber-950/20 border-amber-500/30 text-amber-200'
                : 'bg-rose-950/20 border-rose-500/40 text-rose-200'
            }`}>
              <div className="font-semibold flex items-center gap-1.5 font-mono">
                <span>{previewData.is_executable ? '⚠️ Compatibility Diagnostics' : '🛑 Execution Invariant Error'}</span>
              </div>
              <ul className="list-disc list-inside space-y-0.5 text-[11px] font-sans">
                {previewData.validation_warnings.map((warn, i) => (
                  <li key={i}>{warn}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Output Mode Switcher */}
          <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-900/60">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setActiveTab('table')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    activeTab === 'table'
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800'
                  }`}
                >
                  <TableIcon className="w-3.5 h-3.5" /> Pipeline Configurations ({previewData?.sample_configurations.length ?? 0})
                </button>
                <button
                  onClick={() => setActiveTab('yaml')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    activeTab === 'yaml'
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : 'text-slate-400 hover:text-white hover:bg-slate-800'
                  }`}
                >
                  <FileCode className="w-3.5 h-3.5" /> Specification YAML
                </button>
              </div>

              {activeTab === 'yaml' && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleCopyYaml}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
                  >
                    {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copied ? 'Copied' : 'Copy'}
                  </button>
                  <button
                    onClick={handleDownloadYaml}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded bg-indigo-600 hover:bg-indigo-500 text-white transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" /> Download
                  </button>
                </div>
              )}
            </div>

            {/* Content Area */}
            <div className="p-4">
              {activeTab === 'table' && (
                <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-slate-400 font-mono">
                      <tr>
                        <th className="py-2 px-3">#</th>
                        <th className="py-2 px-3">Chunking</th>
                        <th className="py-2 px-3">Size / Overlap</th>
                        <th className="py-2 px-3">Embedding</th>
                        <th className="py-2 px-3">Retrieval (k)</th>
                        <th className="py-2 px-3">Reranker</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                      {(previewData?.sample_configurations || []).map((cfg) => (
                        <tr key={cfg.index} className="hover:bg-slate-800/40 text-slate-300">
                          <td className="py-2 px-3 text-slate-500">{cfg.index + 1}</td>
                          <td className="py-2 px-3 text-blue-300 font-medium">{cfg.chunking.strategy}</td>
                          <td className="py-2 px-3 text-slate-400">
                            {cfg.chunking.chunk_size} / {cfg.chunking.chunk_overlap}
                          </td>
                          <td className="py-2 px-3 text-emerald-300 truncate max-w-[140px]" title={cfg.embedding.model_name}>
                            {cfg.embedding.model_name.split('/').pop()}
                          </td>
                          <td className="py-2 px-3">
                            <span className="px-1.5 py-0.5 rounded bg-purple-900/30 text-purple-300 border border-purple-800/50">
                              {cfg.retrieval.strategy} (k={cfg.retrieval.top_k})
                            </span>
                          </td>
                          <td className="py-2 px-3 text-amber-300/90 truncate max-w-[140px]">
                            {cfg.reranker.strategy.split('/').pop()}
                          </td>
                        </tr>
                      ))}
                      {(previewData?.sample_configurations.length ?? 0) === 0 && (
                        <tr>
                          <td colSpan={6} className="py-8 text-center text-slate-500">
                            No configurations available. Select at least one option per dimension.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                  {(previewData?.total_combinations ?? 0) > 30 && (
                    <div className="p-2 text-center text-xs text-slate-500 bg-slate-950/60 border-t border-slate-800/80">
                      Showing first 30 of {previewData?.total_combinations} factorial configurations
                    </div>
                  )}
                </div>
              )}

              {activeTab === 'yaml' && (
                <div className="relative">
                  <pre className="p-4 bg-slate-950 rounded-lg text-emerald-300 font-mono text-xs overflow-x-auto max-h-[460px] overflow-y-auto leading-relaxed border border-slate-800">
                    {yamlContent}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
