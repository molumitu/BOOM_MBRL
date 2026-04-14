#!/bin/bash
# Script to run all plotting scripts in plot_explore folder

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Running all plotting scripts in: $SCRIPT_DIR"
echo "========================================================================"

# Array of Python scripts to run in order
scripts=(
    "plot_explore.py"
    "plot_optimal_gaussian.py"
    "plot_optimal_flow.py"
    "plot_plan_gaussian.py"
    "plot_plan_flow.py"
    "plot_train_gaussian.py"
    "plot_train_flow.py"
)

# Counter for successful runs
success_count=0
fail_count=0

# Run each script
for script in "${scripts[@]}"; do
    script_path="$SCRIPT_DIR/$script"

    echo ""
    echo "========================================================================"
    echo "Running: $script"
    echo "========================================================================"

    if [ -f "$script_path" ]; then
        if python "$script_path"; then
            echo "✓ Successfully completed: $script"
            ((success_count++))
        else
            echo "✗ Failed to run: $script"
            ((fail_count++))
        fi
    else
        echo "✗ Script not found: $script_path"
        ((fail_count++))
    fi
done

echo ""
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo "Total scripts: ${#scripts[@]}"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo "========================================================================"

# Exit with error code if any script failed
if [ $fail_count -gt 0 ]; then
    exit 1
fi

exit 0
