

# Copy-on-Write Memory Analysis in Machine Learning Systems

## Project Overview

This project investigates the memory efficiency of Copy-on-Write (CoW) in multiprocess machine learning environments.

It combines operating system concepts, machine learning, and experimental performance analysis to evaluate how memory sharing affects resource consumption and execution overhead.

## Research Objectives

- Investigate how Copy-on-Write enables physical memory sharing between processes.
- Compare Copy-on-Write with independent memory allocation.
- Analyze the impact of memory modification on memory consumption.
- Evaluate scalability with different worker counts and model sizes.
- Investigate selective modification by individual workers.
- Experimentally evaluate the theoretical break-even point.
- Identify the advantages, limitations, and performance trade-offs of CoW.

## Theoretical Background

Copy-on-Write is an operating system memory management technique that allows multiple processes to share physical memory pages until a write operation requires a private copy.

In Linux, processes created through `fork()` can initially share physical memory pages. When a process modifies a shared private mapping, the operating system can create a private copy to preserve process isolation.

The experimental implementation uses Python's `os.fork()` interface directly.

### Theoretical Memory Model

Let:

- `S` = resident model size
- `W` = total number of workers
- `Wm` = number of modifying workers
- `p` = fraction of model pages modified by each modifying worker

The ideal model-specific memory consumption is:

```text
Independent: M = W × S
Copy-on-Write: M = S × (1 + Wm × p)
```

These equations assume resident model pages, a retained original model in the CoW parent, and private copies of modified shared pages.

When all workers modify the model, the theoretical break-even point is:

```text
p = 1 - 1/W
```

For four workers, the theoretical break-even point is 75%.

Total process memory additionally includes non-model overhead, which may differ between strategies.

## Experimental Methodology

Two memory allocation strategies are compared:

- **Independent Allocation:** Each worker maintains a separate model buffer.
- **Copy-on-Write:** Workers inherit a shared model buffer from the parent process.

Each experiment consists of three phases:

1. Initialization
2. Read-only access
3. Controlled memory modification

Each configuration is repeated five times for both strategies.

The complete experimental design consists of:

- 8 experimental configurations
- 40 comparative trials
- 80 strategy runs
- 240 phase-level measurements

### Evaluation Metrics

- Model-specific Proportional Set Size (PSS)
- Total process PSS
- Resident Set Size (RSS)
- Private Dirty Memory
- Minor Page Faults
- Worker Execution Time
- Predicted versus measured model PSS

PSS is the primary memory metric because RSS can count shared physical pages multiple times.

Model-specific PSS is measured using the corresponding virtual memory area (VMA), allowing model memory to be distinguished from other process memory.

## Technologies

- Python 3.14.4
- Linux / Ubuntu
- Windows Subsystem for Linux 2 (WSL2)
- Linux kernel 6.18.58.1-microsoft-standard-WSL2
- NumPy
- Scikit-learn
- Matplotlib
- GitHub

## Machine Learning Models

Both models use an MLPClassifier trained on the Scikit-learn Digits dataset.

| Parameter | Small Model | Large Model |
|---|---:|---:|
| Hidden layer width | 1024 | 4096 |
| Trainable parameters | 1,126,410 | 17,088,522 |
| Model size | 4.30 MiB | 65.19 MiB |
| Test accuracy | 98.22% | 98.22% |
| Data type | Float32 | Float32 |

The system uses a page size of 4096 bytes.

## Experimental Results

Eight experimental configurations were evaluated.

The following table presents mean model-specific PSS after the modification phase.

| Experiment | Workers | Modifying Workers | Modification | Independent PSS | CoW PSS | Memory Saving |
|---|---:|---:|---:|---:|---:|---:|
| E1 — Baseline | 4 | 4 | 25% | 17.203 MiB | 8.594 MiB | 50.05% |
| E2 — Read-only | 4 | 0 | 0% | 17.203 MiB | 4.297 MiB | 75.02% |
| E3 — Moderate write | 4 | 4 | 50% | 17.203 MiB | 12.891 MiB | 25.07% |
| E4 — Full write | 4 | 4 | 100% | 17.203 MiB | 21.504 MiB | -25.00% |
| E5 — Worker scalability | 8 | 8 | 25% | 34.406 MiB | 12.894 MiB | 62.53% |
| E6 — Large model | 4 | 4 | 25% | 260.766 MiB | 130.376 MiB | 50.00% |
| E7 — Selected workers | 4 | 2 | 50% | 17.203 MiB | 8.596 MiB | 50.03% |
| E8 — Break-even | 4 | 4 | 75% | 17.203 MiB | 17.202 MiB | 0.006% |

Modification represents the fraction of model pages modified by each modifying worker.

Negative memory savings indicate that CoW consumes more model memory than independent allocation.

### Selected Workers Experiment

E7 investigates selective modification, where only two of four workers modify 50% of their inherited model pages.

The theoretical model predicts:

```text
M = S × (1 + 2 × 0.50)
M = 2S
```

The measured CoW model PSS was approximately 8.596 MiB, compared with 17.203 MiB for independent allocation.

This represents approximately 50.03% model memory savings and supports the generalized theoretical memory model.

### Break-even Validation

E8 investigates the theoretical break-even point of 75% modification with four workers.

```text
Independent: M = 4S
Copy-on-Write: M = S × (1 + 4 × 0.75) = 4S
```

The measured model PSS values were:

- Independent allocation: 17.203 MiB
- Copy-on-Write: 17.202 MiB

The nearly identical measurements support the theoretical break-even prediction across five repeated trials.

This finding applies specifically to model memory consumption, not necessarily total process memory or execution time.

### Predicted vs. Measured Model PSS

| Experiment | Predicted CoW PSS | Measured CoW PSS | Approximate Deviation |
|---|---:|---:|---:|
| E7 — Selected workers | 8.598 MiB | 8.596 MiB | -2 KiB |
| E8 — Break-even | 17.207 MiB | 17.202 MiB | -5 KiB |

The small deviations demonstrate close agreement between the page-level memory prediction and measured model-specific PSS.

Page granularity and PSS accounting must be considered when interpreting these differences.

### Minor Page Fault Analysis

| Experiment | Page Writes | Mean CoW Minor Faults | Difference |
|---|---:|---:|---:|
| E7 — Selected workers | 1,100 | 1,107 | +7 |
| E8 — Break-even | 3,304 | 3,307 | +3 |

Minor page fault counts closely follow the number of controlled page writes but are not exact measurements of CoW page-copy operations.

Additional minor faults may arise from other memory operations. Their precise causes are not isolated by this experiment.

## Key Findings

- Copy-on-Write reduces model memory consumption by approximately 75% under read-only workloads with four workers.
- Under ideal read-only conditions, the theoretical model predicts 87.5% model memory savings with eight workers.
- Increasing the proportion of modified pages reduces memory savings.
- Larger models increase absolute memory savings, while relative savings remain approximately constant under equivalent worker counts and modification ratios.
- Selective worker modification preserves memory sharing among non-modifying workers.
- The theoretical 75% break-even point is experimentally supported for four workers.
- Write operations introduce additional page-copying and execution overhead.
- Model-specific memory savings differ from total process memory savings.
- Under full modification, retaining the original model in the parent process can make CoW less memory-efficient than independent allocation.

## Repository Structure

| File or Prefix | Description |
|---|---|
| `cow_ml_memory.py` | Experimental implementation |
| `requirements.txt` | Python dependencies |
| `cow_results_*` | Baseline experiment |
| `cow_readonly_*` | Read-only workload |
| `cow_write50_*` | Moderate write workload |
| `cow_write100_*` | Full write workload |
| `cow_workers8_*` | Worker scalability |
| `cow_large_model_*` | Model size scalability |
| `cow_selected_*` | Selected workers experiment |
| `cow_p75_*` | Break-even validation |

Each experiment includes:

- Raw CSV measurements
- Summary CSV statistics
- JSON metadata

## Installation and Usage

A Linux environment or WSL2 is required.

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Baseline Experiment

```bash
python cow_ml_memory.py --strategy both --workers 4 --modify 0.25 --repeats 5 --output baseline_demo
```

### Large Model Experiment

```bash
python cow_ml_memory.py --strategy both --width 4096 --workers 4 --modify 0.25 --repeats 5 --output large_model_demo
```

### Selected Workers Experiment

```bash
python cow_ml_memory.py --strategy both --workers 4 --modify-workers 2 --modify 0.50 --width 1024 --repeats 5 --output selected_demo
```

### Break-even Experiment

```bash
python cow_ml_memory.py --strategy both --workers 4 --modify-workers 4 --modify 0.75 --width 1024 --repeats 5 --output break_even_demo
```

Separate output names are used to preserve the original experimental data.

To view all available parameters:

```bash
python cow_ml_memory.py --help
```

## Limitations

- Experiments were conducted in a WSL2 Linux environment.
- Results concern CPU memory rather than GPU memory.
- Controlled page-access operations do not represent end-to-end ML inference or training performance.
- Minor page faults are not an exact measurement of CoW page copies.
- Model-specific memory savings differ from total process memory savings.
- Non-model memory overhead may vary between allocation strategies.
- Execution time may be influenced by system scheduling and environmental conditions.
- The 75% break-even point was evaluated specifically for four workers.
- Results are limited to the tested models, worker configurations, and experimental environment.

## Conclusion

Copy-on-Write provides substantial model memory savings in read-intensive multiprocess machine learning environments.

However, its effectiveness depends on worker count, model size, page modification ratio, the number of modifying workers, and process architecture.

The selected workers experiment supports the generalized theoretical memory model, while the break-even experiment confirms that the model-memory advantage becomes negligible at approximately 75% modification with four workers.

The results demonstrate a fundamental trade-off between memory efficiency and write-related execution overhead.

## References

- [Operating Systems: Three Easy Pieces](https://pages.cs.wisc.edu/~remzi/OSTEP/)
- [MIT 6.1810 — Copy-on-Write Fork for xv6](https://pdos.csail.mit.edu/6.1810/2023/labs/cow.html)
- [Linux Kernel Documentation — Page Tables](https://docs.kernel.org/mm/page_tables.html)
- [Linux Manual — fork(2)](https://man7.org/linux/man-pages/man2/fork.2.html)
- [Linux Manual — proc_pid_smaps(5)](https://man7.org/linux/man-pages/man5/proc_pid_smaps.5.html)
- [Linux Manual — getrusage(2)](https://man7.org/linux/man-pages/man2/getrusage.2.html)
- [Python Documentation — os.fork()](https://docs.python.org/3/library/os.html#os.fork)
- [Stanford CS111 — Virtual Memory](https://web.stanford.edu/~ouster/cs111-spring24/lectures/vm/)

## Contributors

This project is a collaborative academic research project.

## Research Areas

Operating Systems | Machine Learning Systems | Memory Management | Performance Engineering | Experimental Computer Science

