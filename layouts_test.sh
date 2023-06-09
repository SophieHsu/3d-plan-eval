cleanup() {
    # Kill the Python process if it is running
    if [ -n "$python_pid" ]; then
        echo "Killing Python process..."
        kill "$python_pid"
        wait "$python_pid" 2>/dev/null
        python_pid=""
    fi
    echo "Exiting..."
    exit 1
}

trap 'cleanup' SIGINT

for ((number=3; number<=60; number++))
do
    file_path="kitchen_layouts_grid_text/kitchen${number}.txt"

    if [ "$kill_requested" = true ]; then
        cleanup
    fi

    # Run the Python command in the background and store the process ID
    python 3d_plan_eval_main.py -k "$file_path" &
    python_pid=$!

    # Wait for user input to kill the process or exit
    echo "Press X to kill the process, C to exit, or any other key to continue..."
    read -n 1 -s input

    if [ "$input" = "X" ] || [ "$input" = "x" ]; then
        kill_requested=true
        cleanup
    elif [ "$input" = "C" ] || [ "$input" = "c" ]; then
        cleanup
    fi

    # Check if the process is still running
    if ps -p "$python_pid" > /dev/null; then
        echo "Process is still running..."
        wait "$python_pid"
    fi

    log_file="test_logs/kitchen${number}_log.txt"

    if [ -f "$log_file" ]; then
        last_line=$(tail -n 1 "$log_file")

        if [[ "$last_line" == "success" ]]; then
            ((success_count++))
        fi

        ((total_files++))
    fi
done

success_ratio=$(bc -l <<< "$success_count / $total_files")

echo "Success ratio: $success_count / $total_files = $success_ratio"