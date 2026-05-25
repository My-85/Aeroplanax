"""Maneuver Demo Library — master config listing all demos to render."""

MANEUVER_DEMOS = [
    # ─── Horizontal Turns ───
    {"name": "circle_R5000_right", "traj": "level_circle",
     "params": {"radius": 5000, "direction": 1, "n_points": 60},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 1500,
     "expected": "success", "category": "horizontal_turn"},
    {"name": "circle_R5000_left", "traj": "level_circle",
     "params": {"radius": 5000, "direction": -1, "n_points": 60},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 1500,
     "expected": "success", "category": "horizontal_turn"},
    {"name": "circle_R3000_right", "traj": "level_circle",
     "params": {"radius": 3000, "direction": 1, "n_points": 60},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 1500,
     "expected": "success", "category": "horizontal_turn"},
    {"name": "circle_R3000_left", "traj": "level_circle",
     "params": {"radius": 3000, "direction": -1, "n_points": 60},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 1500,
     "expected": "success", "category": "horizontal_turn"},

    # ─── Horizontal Complex ───
    {"name": "s_curve_A3000", "traj": "s_curve",
     "params": {"amplitude": 3000, "half_period": 10000, "n_points": 80},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 1500,
     "expected": "success", "category": "horizontal_complex"},
    {"name": "figure8_R5000", "traj": "figure_eight",
     "params": {"radius": 5000, "n_points": 100},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 2000,
     "expected": "success", "category": "horizontal_complex"},

    # ─── Mild 3D ───
    {"name": "climb_1000m", "traj": "mild_climb",
     "params": {"length": 15000, "delta_alt": 1000, "n_points": 30},
     "planner": "pure_pursuit", "lookahead": 500, "reach_r": 200, "max_steps": 800,
     "expected": "success", "category": "mild_3d"},
    {"name": "descent_1000m", "traj": "mild_climb",
     "params": {"length": 15000, "delta_alt": -1000, "n_points": 30},
     "planner": "pure_pursuit", "lookahead": 500, "reach_r": 200, "max_steps": 800,
     "expected": "success", "category": "mild_3d"},
    {"name": "climb_2000m", "traj": "mild_climb",
     "params": {"length": 15000, "delta_alt": 2000, "n_points": 30},
     "planner": "pure_pursuit", "lookahead": 500, "reach_r": 200, "max_steps": 800,
     "expected": "success", "category": "mild_3d"},

    # ─── Large-radius Vertical ───
    {"name": "pullup_15deg_R8000", "traj": "vertical_pullup_arc",
     "params": {"radius": 8000, "arc_angle_deg": 15, "n_points": 40},
     "planner": "pure_pursuit", "lookahead": 500, "reach_r": 200, "max_steps": 800,
     "expected": "success", "category": "large_radius_vertical"},
    {"name": "pullup_30deg_R8000", "traj": "vertical_pullup_arc",
     "params": {"radius": 8000, "arc_angle_deg": 30, "n_points": 40},
     "planner": "pure_pursuit", "lookahead": 500, "reach_r": 200, "max_steps": 1200,
     "expected": "success", "category": "large_radius_vertical"},

    # ─── Mild 3D Complex (test) ───
    {"name": "helix_R8000_climb1000", "traj": "helix",
     "params": {"radius": 8000, "turns": 1.0, "delta_alt": 1000, "n_points": 120, "direction": 1},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 2000,
     "expected": "test", "category": "mild_3d_complex"},
    {"name": "climbing_fig8_R5000_dAlt1000", "traj": "climbing_figure_eight",
     "params": {"radius": 5000, "delta_alt": 1000, "n_points": 120},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 2000,
     "expected": "test", "category": "mild_3d_complex"},
    {"name": "climbing_s_A3000_dAlt1000", "traj": "climbing_s_curve",
     "params": {"amplitude": 3000, "half_period": 10000, "delta_alt": 1000, "n_points": 100},
     "planner": "pure_pursuit", "lookahead": 1000, "reach_r": 500, "max_steps": 1500,
     "expected": "test", "category": "mild_3d_complex"},
]
