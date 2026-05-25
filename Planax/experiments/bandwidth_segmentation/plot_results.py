"""
Plot paper-ready figures from adaptive segmentation experiment results.

Reads outputs/adaptive_segmentation/YYYYMMDD_HHMMSS/ and generates:
  1. Reference vs actual trajectory (all methods overlay)
  2. Waypoint distribution comparison
  3. Cross-track error over time
  4. Actuator commands over time
  5. Actuator saturation rate bar plot
  6. Method comparison summary table
  7. Waypoint count vs CTE / saturation / smoothness trade-off

Usage:
    conda activate aeroplanax
    python experiments/bandwidth_segmentation/plot_results.py outputs/adaptive_segmentation/YYYYMMDD_HHMMSS
"""

import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


COLORS = {
    "uniform": "tab:blue",
    "curvature": "tab:orange",
    "rdp": "tab:green",
    "dp_no_bandwidth": "tab:red",
    "dp_with_bandwidth": "tab:purple",
}

MARKERS = {
    "uniform": "o", "curvature": "s", "rdp": "^",
    "dp_no_bandwidth": "D", "dp_with_bandwidth": "P",
}


def load_experiment(result_dir: str):
    """Load all data from an experiment directory."""
    result_path = Path(result_dir)
    data = {"trajectories": {}}

    # Load summary
    summary_path = result_path / "summary.csv"
    if summary_path.exists():
        import csv
        with open(summary_path) as f:
            data["summary"] = list(csv.DictReader(f))

    # Load per-trajectory data
    for traj_dir in sorted(result_path.iterdir()):
        if not traj_dir.is_dir():
            continue
        traj_name = traj_dir.name
        traj_data = {"methods": {}}

        ref_path = traj_dir / "reference.npz"
        if ref_path.exists():
            ref = np.load(ref_path)
            traj_data["reference"] = ref["traj"]

        for npz_path in sorted(traj_dir.glob("*.npz")):
            if npz_path.name == "reference.npz":
                continue
            method_name = npz_path.stem
            d = np.load(npz_path, allow_pickle=True)
            traj_data["methods"][method_name] = {
                "actual_traj": d.get("actual_traj", None),
                "waypoints": d.get("waypoints", None),
                "actions": d.get("actions", None),
                "t": d.get("t", None),
            }

        # Also load metrics
        for json_path in sorted(traj_dir.glob("*_metrics.json")):
            method_name = json_path.stem.replace("_metrics", "")
            with open(json_path) as f:
                if method_name in traj_data["methods"]:
                    traj_data["methods"][method_name]["metrics"] = json.load(f)

        data["trajectories"][traj_name] = traj_data

    return data


def _method_base(method_name: str) -> str:
    """Extract base method name."""
    for prefix in ["dp_with_bandwidth", "dp_no_bandwidth", "uniform", "curvature", "rdp"]:
        if method_name.startswith(prefix):
            return prefix
    return method_name


def plot_trajectory_overlay(data, traj_name: str, output_dir: str):
    """Figure 1: Reference vs actual trajectory with waypoints."""
    if traj_name not in data["trajectories"]:
        return
    tdata = data["trajectories"][traj_name]
    ref = tdata.get("reference")
    if ref is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Trajectory Tracking: {traj_name}", fontsize=14)

    # Top-down view
    ax = axes[0, 0]
    ax.plot(ref[:, 1], ref[:, 0], "k-", lw=1.5, alpha=0.5, label="Reference")
    for mname, md in tdata["methods"].items():
        base = _method_base(mname)
        actual = md.get("actual_traj")
        wps = md.get("waypoints")
        if actual is not None:
            ax.plot(actual[:, 1], actual[:, 0], lw=0.8, alpha=0.6, color=COLORS.get(base, "gray"))
        if wps is not None and base in ["dp_with_bandwidth", "uniform"]:
            ax.scatter(wps[:, 1], wps[:, 0], s=15, marker=MARKERS.get(base, "o"),
                       color=COLORS.get(base, "gray"), label=f"{mname}")
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Top-down trajectory")
    ax.legend(fontsize=6); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # Altitude over time
    ax = axes[0, 1]
    for mname, md in tdata["methods"].items():
        actual = md.get("actual_traj")
        t_arr = md.get("t")
        if actual is not None and t_arr is not None and len(t_arr) == len(actual):
            ax.plot(t_arr, actual[:, 2], lw=0.8, alpha=0.5, color=COLORS.get(_method_base(mname), "gray"))
    ax.axhline(y=ref[0, 2], color="k", ls="--", lw=0.8, label="Target altitude")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude profile"); ax.grid(True, alpha=0.3)

    # Cross-track error over time
    ax = axes[0, 2]
    for mname, md in tdata["methods"].items():
        actual = md.get("actual_traj")
        t_arr = md.get("t")
        if actual is not None and t_arr is not None and ref is not None:
            cte = np.array([np.min(np.linalg.norm(ref - actual[i], axis=1))
                           for i in range(len(actual))])
            ax.plot(t_arr[:len(cte)], cte, lw=0.8, alpha=0.6,
                    color=COLORS.get(_method_base(mname), "gray"), label=mname)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cross-track error (m)")
    ax.set_title("Cross-track error over time"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5)

    # Waypoint distribution (top-down)
    ax = axes[1, 0]
    ax.plot(ref[:, 1], ref[:, 0], "k-", lw=1, alpha=0.3)
    for i, (mname, md) in enumerate(tdata["methods"].items()):
        base = _method_base(mname)
        wps = md.get("waypoints")
        if wps is not None:
            ax.scatter(wps[:, 1], wps[:, 0], s=20, marker=MARKERS.get(base, "o"),
                       color=COLORS.get(base, "gray"), label=f"{mname} ({len(wps)} pts)", alpha=0.7)
    ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
    ax.set_title("Waypoint distributions"); ax.set_aspect("equal")
    ax.legend(fontsize=5); ax.grid(True, alpha=0.3)

    # Actuator commands
    ax = axes[1, 1]
    for mname, md in tdata["methods"].items():
        base = _method_base(mname)
        actions = md.get("actions")
        t_arr = md.get("t")
        if actions is not None and t_arr is not None and base in ["dp_with_bandwidth", "dp_no_bandwidth", "uniform"]:
            ele_norm = actions[:, 1].astype(float) * 2.0 / 40.0 - 1.0
            ax.plot(t_arr[:len(ele_norm)], ele_norm, lw=0.6, alpha=0.4,
                    color=COLORS.get(base, "gray"))
    ax.axhline(y=1.0, color="r", ls=":", lw=0.5); ax.axhline(y=-1.0, color="r", ls=":", lw=0.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Elevator (norm)")
    ax.set_title("Elevator commands"); ax.grid(True, alpha=0.3)

    # Saturation rate bar plot
    ax = axes[1, 2]
    method_names = []
    sat_rates = []
    colors_bar = []
    for mname, md in tdata["methods"].items():
        metrics = md.get("metrics", {})
        sat = metrics.get("actuator_total_saturation_rate", 0)
        method_names.append(mname.replace("_", "\n"))
        sat_rates.append(float(sat))
        colors_bar.append(COLORS.get(_method_base(mname), "gray"))
    bars = ax.bar(range(len(method_names)), sat_rates, color=colors_bar)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, fontsize=6, rotation=45)
    ax.set_ylabel("Saturation rate")
    ax.set_title("Actuator saturation"); ax.grid(True, alpha=0.3, axis="y")

    # 3D trajectory
    ax = axes[2, 0]
    ax.remove(); ax = fig.add_subplot(2, 3, 4, projection="3d")
    ax.plot(ref[:, 1], ref[:, 0], ref[:, 2], "k-", lw=1.5, alpha=0.4, label="Ref")
    for mname, md in tdata["methods"].items():
        actual = md.get("actual_traj")
        base = _method_base(mname)
        if actual is not None and base in ["dp_with_bandwidth", "uniform_N40"]:
            ax.plot(actual[:, 1], actual[:, 0], actual[:, 2], lw=0.8, alpha=0.7,
                    color=COLORS.get(base, "gray"), label=mname)
    ax.set_xlabel("East"); ax.set_ylabel("North"); ax.set_zlabel("Alt")
    ax.set_title("3D view"); ax.legend(fontsize=5)

    # Speed profile
    ax = axes[2, 1]
    for mname, md in tdata["methods"].items():
        state = md.get("state") or {}
        t_arr_val = md.get("t")
        airspeed = state.get("airspeed", [])
        if t_arr_val is not None and len(airspeed) > 0:
            ax.plot(t_arr_val[:len(airspeed)], airspeed, lw=0.8, alpha=0.5,
                    color=COLORS.get(_method_base(mname), "gray"))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Airspeed (m/s)")
    ax.set_title("Airspeed"); ax.grid(True, alpha=0.3)

    # Metrics text
    ax = axes[2, 2]
    ax.axis("off")
    lines = ["=== Metrics Summary ===", ""]
    for mname, md in tdata["methods"].items():
        m = md.get("metrics", {})
        lines.append(f"{mname}:")
        lines.append(f"  CTE_rms={m.get('cross_track_error_continuous_rms_m', '?'):.1f}m")
        lines.append(f"  Sat={m.get('actuator_total_saturation_rate', '?'):.3f}")
        lines.append(f"  WP_reached={m.get('waypoints_reached', '?')}")
        lines.append("")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace")

    fig.savefig(os.path.join(output_dir, f"trajectory_{traj_name}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(data, output_dir: str):
    """Figure 2: Waypoint count vs performance trade-off."""
    if "summary" not in data or not data["summary"]:
        return

    summary = data["summary"]
    methods_unique = sorted(set(r["method"] for r in summary))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Waypoint Count vs Performance Trade-off", fontsize=14)

    for ax, metric, ylabel in [
        (axes[0, 0], "cross_track_rms", "Cross-track error RMS (m)"),
        (axes[0, 1], "total_saturation_rate", "Actuator saturation rate"),
        (axes[1, 0], "actuator_smoothness", "Actuator smoothness (ΣΔu²)"),
    ]:
        for traj in sorted(set(r["trajectory"] for r in summary)):
            rows = [r for r in summary if r["trajectory"] == traj]
            xs = [float(r.get("N_waypoints", 0)) for r in rows]
            ys = [float(r.get(metric, 0)) for r in rows]
            ax.scatter(xs, ys, alpha=0.7, label=traj)
        ax.set_xlabel("Number of waypoints"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # Per-method summary
    ax = axes[1, 1]
    ax.axis("off")
    lines = ["=== Per-Method Aggregate ===", ""]
    for method in methods_unique:
        rows = [r for r in summary if r["method"] == method]
        if rows:
            cte_vals = [float(r.get("cross_track_rms", 0)) for r in rows if r.get("cross_track_rms")]
            sat_vals = [float(r.get("total_saturation_rate", 0)) for r in rows if r.get("total_saturation_rate")]
            lines.append(f"{method}:")
            lines.append(f"  Avg CTE_rms = {np.mean(cte_vals):.1f}m" if cte_vals else "  N/A")
            lines.append(f"  Avg Sat = {np.mean(sat_vals):.3f}" if sat_vals else "  N/A")
            lines.append("")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=7,
            verticalalignment="top", fontfamily="monospace")

    fig.savefig(os.path.join(output_dir, "tradeoff_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", help="Path to experiment output directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory for plots")
    args = parser.parse_args()

    result_dir = args.result_dir
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(result_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    data = load_experiment(result_dir)
    print(f"Loaded {len(data['trajectories'])} trajectories")

    for traj_name in data["trajectories"]:
        print(f"  Plotting {traj_name}...")
        plot_trajectory_overlay(data, traj_name, output_dir)

    if data.get("summary"):
        print(f"  Plotting trade-off summary...")
        plot_tradeoff(data, output_dir)

    print(f"\nPlots saved to: {output_dir}")
    for f in sorted(Path(output_dir).glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
