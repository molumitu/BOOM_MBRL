# Four-Goal MPPI Implementation Task

## Task Description

Create a `four_goal_mppi.py` file that implements Model Predictive Path Integral (MPPI) optimization with warm-start initialization from filtered directed random trajectories for the four-goal navigation environment.

## Core Requirements

### Two-Stage Strategy

1. **Stage 1 - Candidate Generation & Filtering**:
   - Generate a batch of directed random trajectories
   - Filter trajectories that get close to goals or successfully reach them
   
2. **Stage 2 - MPPI Optimization**:
   - Use the filtered trajectory's action sequence as MPPI initial value
   - Perform MPPI optimization to refine the trajectory

### Filtering Criteria

A trajectory passes the filter if:
- Final position distance to any goal < `distance_threshold` (e.g., 0.3)
- OR trajectory successfully reaches a goal

### Experiment Setup

- Run `N_EXPERIMENTS` independent experiments
- Save the final optimized trajectory for each experiment

## Reference Files

### `four_goal_nav_env.py` - Four-Goal Navigation Environment

- **Class**: `FourGoalNavigationEnv`
- **Four Goals**:
  - `top_left`: (-1, 1)
  - `top_right`: (1, 1)
  - `bottom_left`: (-1, -1)
  - `bottom_right`: (1, -1)
- **Action Space**: Direction angle theta ∈ [-π, π]
- **State Space**: Position (x, y) ∈ [-1, 1]²

### `four_goal_random_traj.py` - Directed Random Trajectory Generation

- `generate_directed_random_trajectory()`: Generate a single directed random trajectory
- `compute_optimal_angle()`: Compute optimal angle towards goal
- `noise_std` parameter controls randomness

### `visualize_mppi.py` - MPPI Implementation (Two-Goal Environment)

- `rollout_trajectory()`: Execute action sequence and return trajectory
- `mppi_optimize()`: Core MPPI optimization logic
- `run_experiments()`: Run multiple independent experiments
- `plot_best_trajectories()`: Visualize best trajectories
- `print_statistics()`: Print statistics
- `save_results_to_file()`: Save results

## Implementation Structure

```python
# Main Functions

def generate_and_filter_trajectories(env_params, num_candidates, distance_threshold, noise_std, rng):
    """
    Generate directed random trajectories and filter those close to goals.
    
    Returns:
        filtered_trajectories: List of filtered trajectories (with action sequences)
        target_goals: Target goal for each trajectory
    """
    pass

def mppi_optimize_with_warm_start(
    env_class, 
    env_params, 
    initial_action_sequence,  # From filtered trajectory
    target_goal,              # Target direction (optional, for biased sampling)
    horizon, 
    num_samples, 
    num_iterations,
    lambda_, 
    noise_sigma, 
    seed
):
    """
    MPPI optimization with warm start (initial action sequence from filtered trajectory).
    """
    pass

def run_four_goal_experiments(env_params, num_experiments, ...):
    """
    Run four-goal MPPI experiments.
    """
    pass

def plot_four_goal_trajectories(results, env_params, save_path):
    """
    Visualize trajectories in four-goal environment.
    """
    pass

def plot_first_action_distribution(results, save_path):
    """
    Plot distribution of first actions.
    
    - X-axis: Action angle in radians
    - Y-axis: Density
    - Include histogram and KDE fitted curve
    
    Four optimal direction reference lines:
    - top_left: 3π/4 (135°)
    - top_right: π/4 (45°)
    - bottom_left: -3π/4 (-135°)
    - bottom_right: -π/4 (-45°)
    """
    pass
```

## Four-Goal Environment Specific Handling

### Goal Color Mapping

| Goal | Position | Color |
|------|----------|-------|
| `top_left` | (-1, 1) | Red |
| `top_right` | (1, 1) | Green |
| `bottom_left` | (-1, -1) | Blue |
| `bottom_right` | (1, -1) | Orange |

### Multi-Modal Handling

- Consider trajectory's target goal during MPPI initialization
- Optionally add bias in action noise sampling

### Visualization

- Group trajectories by target goal
- Legend shows success count for each goal

## Suggested Parameters

```python
ENV_PARAMS = {
    'step_size': 0.10,
    'goal_radius': 0.1,
    'max_steps': 40,
    'step_penalty': -0.01
}

# Stage 1: Trajectory Generation & Filtering
NUM_CANDIDATES = 200        # Number of candidate trajectories
DISTANCE_THRESHOLD = 0.3   # Distance threshold for filtering
NOISE_STD = np.pi / 3      # Random noise standard deviation

# Stage 2: MPPI Optimization
HORIZON = 40               # Planning horizon
NUM_SAMPLES = 128          # Samples per iteration
NUM_ITERATIONS = 50        # MPPI iterations
LAMBDA = 1.0               # Temperature parameter
NOISE_SIGMA = 0.5          # Action noise

N_EXPERIMENTS = 50         # Number of independent experiments
```

## Output Requirements

1. Create timestamped output directory: `figures/MMDD-HHMMSS/`
2. Save contents:
   - Best trajectories plot (`mppi_best_trajectories.png`)
   - Convergence analysis plot (`mppi_convergence_example.png`)
   - **First action distribution plot** (`first_action_distribution.png`)
   - Results data (`mppi_results.npy`)
   - Parameter configurations (`env_params.json`, `mppi_params.json`)

## First Action Distribution Plot Details

```python
def plot_first_action_distribution(results, save_path):
    """
    Plot distribution of first actions across all experiments.
    
    Contents:
    1. Histogram: Show distribution of action angles (bins=36 or more)
    2. KDE curve: Kernel density estimation using scipy.stats.gaussian_kde
    3. Four vertical reference lines: Mark optimal directions for four goals
    
    Four optimal directions:
    - top_left: 3π/4 = 2.356 rad (135°)
    - top_right: π/4 = 0.785 rad (45°)
    - bottom_left: -3π/4 = -2.356 rad (-135°)
    - bottom_right: -π/4 = -0.785 rad (-45°)
    
    Style:
    - Use different colored dashed lines for four optimal directions
    - Legend explaining each reference line's corresponding goal
    - Title includes experiment count and success statistics
    """
```

## Implementation Notes

1. All code should be in English
2. Follow the style and structure of `visualize_mppi.py`
3. Adapt two-goal logic to four-goal scenario
4. Ensure proper environment isolation for parallel rollouts