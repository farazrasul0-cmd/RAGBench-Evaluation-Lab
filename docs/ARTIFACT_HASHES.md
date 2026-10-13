# Cryptographic Artifact Hashes & Verification Manifest

This document records the canonical SHA-256 cryptographic digests for all frozen empirical datasets, reference databases, replication bundles, and benchmark corpora in **RAGBench-Evaluation-Lab**.

External researchers can verify the cryptographic integrity of these files to guarantee that evaluation evidence and statistical baselines have remained immutable and untampered with.

---

## 1. Reference Databases & Statistical Evidence

| File Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| `data/experiments/g_multilingual_reference_studies.db` | `9e46ff96d9b4db3cbf2d1575722fef724172c1a0103639095018bc8865290631` | Authoritative SQLite database storing all 12 factorial RQ3 runs ($N=25$ matched units). |
| `data/experiments/g_multilingual_replication_bundle/statistical_summary.json` | `b30b9daae7d7328e72ed488bdb82c8551422c89d206dd10147ceb34f4679caf0` | Frozen parametric and non-parametric statistical summaries (Wilcoxon, paired $t$, Holm-Bonferroni). |
| `data/experiments/g_multilingual_replication_bundle/manifest.json` | `cf8c7d690ad1243f768c4d87afaf05cdf2a3a94dbf421669d9ca1971bf5e7044` | Replication bundle manifest recording environment, revisions, and execution parameters. |
| `data/experiments/g_multilingual_replication_bundle/hashes.json` | `b1038af6e832135483d5b1c40e589df4d16956736bc6d23bcb892e78543bc588` | Internal bundle file checksums. |
| `data/experiments/g_multilingual_replication_bundle/raw_paired_observations.csv` | `a5fb3162f4e3c3afcf9fef8f99e25e049fdb2ce810ba4b678fd176a677415be1` | Raw paired observation values across all $N=25$ units for independent re-analysis. |
| `data/experiments/g_multilingual_results.json` | `b30b9daae7d7328e72ed488bdb82c8551422c89d206dd10147ceb34f4679caf0` | Consolidated experimental sweep results. |

---

## 2. Benchmark Evaluation Corpora

| File Path | SHA-256 Checksum | Description |
| :--- | :--- | :--- |
| `data/benchmarks/multilingual_bengali_english/eval_qa.jsonl` | `a532245311e3b7ec3ed0914fd5c653425d55d40296bdd0ce18783d7a26aaed31` | Primary bilingual evaluation queries ($N=25$ parallel information units). |
| `data/benchmarks/multilingual_bengali_english/eval_qa_bn_bn.jsonl` | `4ac9cd554e806d01eca44a6a3e9ac86fef63f91892067545ae4be9daa6c9f423` | Bengali Monolingual query-document evaluation set (BN $\to$ BN). |
| `data/benchmarks/multilingual_bengali_english/eval_qa_bn_en.jsonl` | `e55e5b6e043d89121676dd397b122d7a462232283a85f412b40f75c3d239b785` | Bengali-to-English cross-lingual query-document evaluation set (BN $\to$ EN). |
| `data/benchmarks/multilingual_bengali_english/eval_qa_en_bn.jsonl` | `7a16c8d140e6b3207c0220109a79b9ff5696393304e73dfe96bf27c5a4c3230c` | English-to-Bengali cross-lingual query-document evaluation set (EN $\to$ BN). |
| `data/benchmarks/multilingual_bengali_english/eval_qa_en_en.jsonl` | `6cb87b8fc8263fee0c38efd8b475cad3c535c1dc7ad391b4534e28e3b45129d8` | English Monolingual query-document evaluation set (EN $\to$ EN). |
| `data/benchmarks/scifact_ragbench/corpus.jsonl` | `f61ff9c33350f20f8441c224ab40d0a2162cfd3449c2e8e71cf18c0d09ccab08` | SciFact scientific corpus subset (50 claims, 30 evidence abstracts). |
| `data/benchmarks/scifact_ragbench/eval_qa.jsonl` | `900fa87fae010d012c38dee41efa1da01fdcc6e176b44aab70f69fedc5ac10e9` | SciFact evaluation claims and evidence ground truth. |

---

## 3. Verification Instructions

### On Linux / macOS:
```bash
sha256sum data/experiments/g_multilingual_reference_studies.db
sha256sum data/experiments/g_multilingual_replication_bundle/statistical_summary.json
```

### On Windows (PowerShell):
```powershell
Get-FileHash data/experiments/g_multilingual_reference_studies.db -Algorithm SHA256
Get-FileHash data/experiments/g_multilingual_replication_bundle/statistical_summary.json -Algorithm SHA256
```
