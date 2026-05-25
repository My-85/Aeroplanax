# Style Review Before Codex Rewrite

## Current Draft Problems

- The English draft has the right technical ingredients but reads like an AI-generated summary: many paragraphs begin with "We present / We introduce / We show", and several sections are organized as claim lists rather than as robotics questions answered by evidence.
- The abstract is overloaded. It tries to cover the skill, RH-TSO, energy-aware fine-tuning, Euler-vs-quaternion comparison, loop metrics, and failure boundaries in one long sequence.
- The contribution list has only four items and underplays the executable target-stream abstraction as a contribution distinct from RH-TSO.
- The method section is overly enumerative. Observation and reward details are listed before the reader has a crisp view of the system interface and why target streams are the compositional object.
- The results section contains multiple strong numerical claims whose evidence is not present in the paper directory, especially Euler-vs-quaternion superiority and vertical completion percentages.
- Figures are placeholders in comments rather than CoRL-style visual evidence. The draft needs at least schematic placeholder figures that compile and indicate the intended evidence.
- The paper sometimes sounds like a benchmark/platform paper because the simulation substrate is named too prominently. Planax should not appear in the anonymous manuscript.

## Overclaims To Remove Or Soften

- "Quaternion target encoding substantially outperforms Euler-angle encoding" is too strong without a visible table. The safer claim is that quaternion targets avoid representational discontinuities near vertical attitudes.
- The vertical completion-rate table gives exact percentages for 60, 90, 120, and 150 degree arcs, but those numbers are not supported by files in this manuscript directory. The rewrite should mark the detailed table as pending verification or report only the qualitative frontier requested by the prompt.
- Any wording that suggests 180 degree half-loop success, full-loop success, complete aerobatics, or full-envelope maneuvering must be removed. The manuscript should state that 180 degree half-loop is a failure boundary.
- RH-TSO should not be described as globally optimal. The current implementation is grid-optimal only on a deterministic lattice in a low-dimensional target-stream space.
- CEM/MPPI should be framed as a natural extension for higher-dimensional or continuous target-stream spaces, not as better than lattice search in the current experiments.

## Missing Evidence

- Low-level skill tracking numbers need source files or should be presented as placeholders/TODOs.
- Euler-vs-quaternion comparison needs a table, plot, or explicit experiment log before making quantitative claims.
- Loop-quality frontier needs the actual angle-by-angle metrics: CTE, velocity-tangent error, nose-tangent error, wing-plane error, quaternion geodesic error, alpha/beta/G/speed safety, and grade.
- Representative maneuver figures are missing from the manuscript directory.
- The current RH-TSO evidence is solid and should be the main quantitative result: the four-task parameter sweep in `results/short_horizon_target_stream_selection/20260519_122354/`.

## Weak Sections

- Introduction: currently broad and contribution-heavy. It should start from the concrete fixed-wing difficulty: attitude, velocity, lift direction, energy, and actuator bandwidth are coupled, so waypoint tracking is the wrong abstraction.
- Related Work: current entries are too much like paper summaries. The revised section should compare interfaces: perception-to-waypoint, learned tracking policies, learned trajectory generation, simulator infrastructure, and direct RL control.
- Method: should foreground the executable target stream and RH-TSO rollout equation. Detailed PPO and reward implementation should be compact.
- Results: each subsection should answer a specific question. RH-TSO should ask whether target-stream parameters matter; geometry-aware evaluation should ask whether CTE alone is enough.
- Discussion: should be candid about simulation-only validation, verified number gaps, and the 180 degree failure boundary.

## Reference-Paper Positioning Notes

- Deep Drone Racing and Combining Optimal Control and Learning both use inspectable intermediate representations to connect learned components with classical planning/control. This is the closest stylistic precedent for our target-stream interface.
- DATT frames trajectory tracking around infeasible references, actuation limits, and online adaptation. Our paper should similarly treat fixed-wing references as executable only through a closed-loop skill, not as geometry that can be tracked directly.
- Real-Time Generation of Time-Optimal Quadrotor Trajectories frames learned generation around feasibility tests and online speed. Our RH-TSO framing should emphasize closed-loop feasibility under the frozen skill.
- Flightmare is a simulator paper; it should motivate high-throughput robot-learning evaluation but not become the paper's identity.
- Decentralized quadrotor swarms and Soft Multicopter Control are useful for positioning direct learned control and learned dynamics/control interfaces, but they should not dominate the story.

## Chinese Draft Issues

- The Chinese draft is mostly literal translation and inherits the English overclaims.
- Some phrases are unnatural in Chinese academic writing, especially "rollout" and list-like contribution sentences.
- Technical terms should be standardized: 滚动时域目标流优化, 可执行目标流, 四元数条件飞行技能, 作动器级直接控制, 几何感知评估, 横偏误差, 翼面误差, 机头切线误差, 速度切线误差.
- Author, affiliation, email, acknowledgments, and Planax-identifying language must be removed or anonymized.
