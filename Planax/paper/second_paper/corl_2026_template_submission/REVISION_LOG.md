# Revision Log

## Files Revised

- `main.tex`: new primary English manuscript.
- `example.tex`: synchronized to `main.tex` so the old English draft is not left as the apparent working file.
- `example_cn.tex`: rewritten Chinese manuscript, aligned with the revised English structure and claims.
- `example.bib`: corrected reference entries to match the downloaded PDFs.
- `STYLE_REVIEW.md`: internal second-author review before rewriting.
- `TODO_NUMBERS_AND_FIGURES.md`: remaining verification and figure work.
- `compile_log.txt`: compile attempt log.

## Major Structural Changes

- Reframed the paper around fixed-wing maneuvering rather than a benchmark/platform story.
- Shortened and disciplined the abstract around the problem, direct-actuator skill, executable target streams, RH-TSO, verified RH-TSO numbers, and the vertical failure boundary.
- Added a five-part contribution list:
  1. PID-free quaternion-conditioned flight skill.
  2. Executable target-stream abstraction.
  3. Receding-Horizon Target-Stream Optimization.
  4. Energy-aware vertical extension.
  5. Geometry-aware maneuver evaluation.
- Reorganized the method around the system interface, target-stream composition, RH-TSO rollout objective, energy-aware extension, and geometry-aware metrics.
- Rewrote related work as comparative positioning against CoRL-style learning/control interfaces rather than a list of summaries.
- Reworked results into question-driven subsections and centered the quantitative evidence on the verified RH-TSO sweep.

## Claims Removed Or Softened

- Removed the strong claim that quaternion targets quantitatively outperform Euler targets for maneuvers above 60 degrees. The draft now states the representational reason and marks the quantitative ablation as needing verification.
- Removed exact vertical completion-rate percentages that were not supported by files in this manuscript directory.
- Removed any claim that 180 degree half-loop is solved or executable with acceptable quality.
- Removed full-envelope and complete-aerobatics framing.
- Removed Planax/team acknowledgments and other self-identifying submission text.
- Softened RH-TSO optimality: it is grid-optimal only over the current deterministic lattice, not globally optimal.
- Kept CEM/MPPI only as a natural extension for higher-dimensional or continuous target-stream spaces.

## RH-TSO Numbers Added

From `results/short_horizon_target_stream_selection/20260519_122354/`:

| Maneuver | Best (L, vt) | Best CTE_p90 | Default CTE_p90 | Improvement |
|---|---:|---:|---:|---:|
| S-curve | (600, 220) | 918 | 1503 | 39% |
| Figure-eight | (600, 220) | 1017 | 1039 | 2% |
| Helix / mild-3D | (600, 220) | 524 | 691 | 24% |
| 90 degree vertical pull-up | (1500, 280) | 51 | 96 | 47% |

## References Corrected

- Deep Drone Racing: Learning Agile Flight in Dynamic Environments.
- Combining Optimal Control and Learning for Visual Navigation in Novel Environments.
- DATT: Deep Adaptive Trajectory Tracking for Quadrotor Control.
- Real-Time Generation of Time-Optimal Quadrotor Trajectories with Semi-Supervised Seq2Seq Learning.
- Decentralized Control of Quadrotor Swarms with End-to-End Deep Reinforcement Learning.
- Soft Multicopter Control Using Neural Dynamics Identification.
- Flightmare: A Flexible Quadrotor Simulator.

The previous `.bib` file had several mismatched author lists and years relative to the downloaded PDFs; these were corrected from the PDF first pages.

## Compile Status

- `latexmk` is unavailable.
- `pdflatex` is unavailable.
- `xelatex`, `lualatex`, `bibtex`, and `tectonic` are also unavailable.
- Therefore the PDF could not be compiled in this environment. See `compile_log.txt`.

## Static Checks Performed

- `main.tex` and `example.tex` are identical.
- Citation keys used in `main.tex` and `example_cn.tex` are present in `example.bib`.
- Manuscript files no longer include author names, affiliations, emails, GitHub links, local paths, or Planax acknowledgments.
- Claim-sensitive search terms were checked; remaining mentions of solved/full-loop concepts are negated limitation statements.
