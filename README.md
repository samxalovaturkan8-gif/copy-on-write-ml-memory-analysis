
# Copy-on-Write Memory Analysis in Machine Learning Systems

## Project Overview

This project investigates the memory efficiency of Copy-on-Write (CoW) in multiprocess machine learning environments.

It combines operating system concepts, machine learning, and experimental performance analysis to evaluate how memory sharing affects resource consumption and execution overhead.

## Research Objectives

- Investigate how Copy-on-Write enables physical memory sharing between processes.
- Compare Copy-on-Write with independent memory allocation.
- Analyze the impact of memory modification on memory consumption.
- Evaluate scalability with different worker counts and model sizes.
- Identify the advantages, limitations, and performance trade-offs of CoW.

## Theoretical Background

Copy-on-Write is an operating system memory management technique that allows multiple processes to share physical memory pages until a write operation requires a private copy.

In Linux, processes created through `fork()` can initially share memory pages. When a process modifies a shared page, the operating system can create a private copy to preserve process isolation.

## Experimental Methodology

Two memory allocation strategies are compared:

- **Independent Allocation:** Each worker maintains a separate model buffer.
- **Copy-on-Write:** Workers inherit a shared model buffer from the parent process.

Each experiment consists of three phases:

1. Initialization
2. Read-only access
3. Controlled memory modification

Each configuration is repeated five times.

### Evaluation Metrics

- Proportional Set Size (PSS)
- Resident Set Size (RSS)
- Private Dirty Memory
- Minor Page Faults
- Worker Execution Time

PSS is the primary memory metric because RSS can count shared physical pages multiple times.

## Technologies

- Python 3.14.4
- Linux / Ubuntu
- Windows Subsystem for Linux 2 (WSL2)
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

## Experimental Results

Six experimental configurations were evaluated, producing 180 phase-level measurements.

The following table presents mean model PSS after the modification phase.

| Experiment | Workers | Modification | Independent PSS | CoW PSS | Memory Saving |
|---|---:|---:|---:|---:|---:|
| Baseline | 4 | 25% | 17.203 MiB | 8.594 MiB | 50.05% |
| Read-only | 4 | 0% | 17.203 MiB | 4.297 MiB | 75.02% |
| Moderate write | 4 | 50% | 17.203 MiB | 12.891 MiB | 25.07% |
| Full write | 4 | 100% | 17.203 MiB | 21.504 MiB | -25.00% |
| Worker scalability | 8 | 25% | 34.406 MiB | 12.894 MiB | 62.53% |
| Large model | 4 | 25% | 260.766 MiB | 130.376 MiB | 50.00% |

## Key Findings

- Copy-on-Write reduces model memory consumption by approximately 75% under read-only workloads with four workers.
- Increasing the number of workers to eight increases read-only model memory savings to approximately 87.5%.
- Increasing the proportion of modified pages reduces memory savings.
- Larger models provide greater absolute memory savings.
- Write operations introduce additional page-copying overhead.
- Under full modification, retaining the original model in the parent process can make CoW less memory-efficient than independent allocation.

## Repository Structure

- `cow_ml_memory.py` — Experimental implementation
- `requirements.txt` — Python dependencies
- `cow_results_*` — Baseline experiment
- `cow_readonly_*` — Read-only workload
- `cow_write50_*` — Moderate write workload
- `cow_write100_*` — Full write workload
- `cow_workers8_*` — Worker scalability
- `cow_large_model_*` — Model size scalability

Each experiment includes raw CSV measurements, summary statistics, and JSON metadata.

## Installation and Usage

A Linux environment or WSL2 is required.

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the baseline experiment:

```bash
python cow_ml_memory.py --workers 4 --modify 0.25 --repeats 5 --output baseline_demo
```

Run the large-model experiment:

```bash
python cow_ml_memory.py --width 4096 --workers 4 --modify 0.25 --repeats 5 --output large_model_demo
```

## Limitations

- Experiments were conducted in a WSL2 Linux environment.
- Results concern CPU memory rather than GPU memory.
- Controlled memory operations do not represent end-to-end ML inference performance.
- Minor page faults are not an exact measurement of CoW page copies.
- Model-specific memory savings differ from total process memory savings.
- The theoretical break-even point was not independently tested.

## Conclusion

Copy-on-Write provides substantial memory savings in read-intensive multiprocess machine learning environments.

However, its effectiveness depends on the number of worker processes, model size, memory modification rate, and process architecture.

The results demonstrate a fundamental trade-off between memory efficiency and write-related execution overhead.

## Contributors

This project is a collaborative academic research project.

## Research Areas

Operating Systems | Machine Learning Systems | Memory Management | Performance Engineering | Experimental Computer Science
