# awq-vs-gptq-int4-marlin-bench

## Context

**Techniques from 2023, reimplemented in 2026.** Era 2: Senior AI Engineer, efficient LLM inference. This repository compares round-to-nearest (RTN), activation-aware AWQ, and Hessian-compensated GPTQ using real packed groupwise INT4 weights and a mixed-precision torch decode path.

## Problem

Weight-only INT4 reduces memory traffic, but independent rounding can disproportionately damage channels with large activations or correlated inputs. AWQ protects activation-salient channels by scaling weights before quantization. GPTQ instead quantizes sequentially and updates not-yet-quantized weights to compensate for calibration loss. A useful comparison must execute the packed representation and measure both model quality and decode throughput, not just report fake-quantized weight error.

## Method

For each output row and input group, asymmetric unsigned INT4 uses

```
scale = (max(W_group) - min(W_group)) / 15
zero  = clamp(round(-min(W_group) / scale), 0, 15)
q     = clamp(round(W_group / scale) + zero, 0, 15)
```

Two nibbles are packed into each `uint8`, low nibble first. Odd input widths are padded only in storage. `Int4Linear` unpacks the nibbles, reconstructs `scale * (q - zero)` in fp32, and multiplies fp32 activations by that matrix. This is actual 4-bit persistent storage with an explicit mixed-precision matmul, not a call to a uniform fake-quantization operator. Constant positive and negative groups have dedicated exact encodings.

**AWQ.** For calibration activations `X`, salience is `s_j = mean_i |X_ij|`. Candidates `s_j ** alpha` are normalized to mean one for `alpha ∈ {0, .25, .5, .75, 1}`. The implementation quantizes `W * s`, evaluates `(X / s) @ dequant(W_q).T`, and selects the candidate with the lowest calibration MSE. The inverse input scale is applied at inference. Alpha zero is the RTN candidate.

**GPTQ.** Calibration estimates `H = 2 XᵀX / n + lambda I`. A damped inverse-Cholesky factor drives sequential column quantization. After quantizing column `i`, each remaining column is updated by

```
W[:, i+1:] -= ((W[:, i] - Wq[:, i]) / Hinv[i,i]) * Hinv[i,i+1:]
```

Group scales and zero points come from the original weights; later quantized values include compensation from earlier columns. This is the Hessian-based error-compensation algorithm, implemented directly on small torch matrices.

The end-to-end model is a learned 32-token, 32-hidden bigram LM with an embedding, RMSNorm, a two-layer MLP residual block, and an output head. Its three linear layers are replaced by packed INT4 modules. Calibration captures their real inputs with forward pre-hooks. Evaluation uses next-token cross entropy; throughput uses autoregressive greedy decoding.

## Results

Measured with Python 3.12.3, torch 2.15.0+cpu, one CPU thread, 160 AdamW steps, 128 training and 128 disjoint evaluation sequences of length 33 from the same three-successor Markov process, group size 16, and seven decode repetitions. Decode used batch 16 for 32 tokens per repetition. Training loss fell from 3.57505 to 1.12816. Raw measurements and configuration are committed in `results/tiny_cpu.json`; the plot is `results/tiny_cpu_pareto.svg`.

| method | held-out perplexity | median greedy tok/s | linear storage bytes |
|---|---:|---:|---:|
| fp32 | **3.168666** | **51,845.18** | 20,864 |
| RTN INT4 | 3.175147 | 12,358.04 | 5,056 |
| AWQ INT4 | 3.184247 | 16,620.84 | 5,056 |
| GPTQ INT4 | 3.175018 | 17,055.97 | 5,056 |

The INT4 representations use **75.7669% fewer linear-layer bytes** than fp32, including scales, zero points, biases, and explicit input-scale vectors; embeddings and activations are excluded. GPTQ is best among the INT4 perplexities by a very small margin, while AWQ is worse than RTN here. This single tiny task does not establish a general ranking.

The CPU packed path is **slower** than fp32: unpacking and dequantization dominate these tiny matrix multiplications. These are real throughput numbers, not a claimed Marlin speedup. The non-dominated points in this run are fp32, GPTQ, and AWQ; RTN is narrowly dominated by GPTQ.

![Measured perplexity and decode throughput](results/tiny_cpu_pareto.svg)

**Marlin and GPU status: pending.** No Marlin kernel, NVIDIA GPU result, large-transformer perplexity, or hardware memory saving is claimed. The reserved Marlin benchmark interface is:

```bash
PYTHONPATH=src python -m awq_gptq.marlin_bench \
  --model-checkpoint /path/to/local/model --dataset /path/to/local/tokens \
  --group-size 128 --batch-size 16 --decode-tokens 128 --repeats 50 \
  --output results/marlin_cuda.json
```

That module is intentionally not present yet; the command records the exact follow-up interface rather than disguising the CPU implementation as Marlin.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest -q
scripts/smoke.sh
scripts/benchmark.sh --smoke --output results/smoke.tmp
scripts/benchmark.sh --output results/tiny_cpu.json
```

On this workspace, use `PYTHON_BIN=/home/pus/.venvs/portfolio/bin/python` for the shell scripts. Rerunning overwrites the committed measurements with current timings. The benchmark has no network, dataset-download, transformers, or GPU dependency.

The 20-test suite covers nibble order, odd widths, independent groups, constant groups, real packed matmul, AWQ salience and inverse scaling, GPTQ compensation and group boundaries, tiny-model training/evaluation/decode, storage accounting, benchmark output, and Pareto selection. Mutation checks replaced independent group scales, disabled AWQ salience, and removed GPTQ's residual update; each made pytest fail. Source was restored and all 20 tests passed again.

## Limitations

- The perplexity task is synthetic and the model is deliberately tiny. It demonstrates end-to-end correctness and measurement, not AWQ/GPTQ quality on Llama-scale checkpoints or natural-language corpora.
- GPTQ applies full trailing-matrix updates for clarity. Production lazy batch/block updates are more efficient; calibration cost is not included in decode tok/s.
- The torch path reconstructs fp32 weights on every invocation. It does not fuse unpacking with a hardware integer tensor-core kernel, and its storage accounting is analytical object-size accounting rather than measured RSS.
- AWQ uses output MSE on captured calibration inputs and a small alpha grid. It does not reproduce every implementation detail of the original AutoAWQ kernels.
- GPU/Marlin support is unimplemented and pending; the command above is a specified follow-up, not an existing runnable benchmark.

## References

- J. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. arXiv:2306.00978.
- E. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323.
- E. Frantar and D. Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* ICML 2023 (related optimal brain compression background). arXiv:2301.00774.
- Marlin: https://github.com/IST-DASLab/marlin (pending GPU kernel comparison).

## License

MIT
