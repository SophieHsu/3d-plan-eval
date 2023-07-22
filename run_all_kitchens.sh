#!/bin/bash

success_count=0
total_files=0
kill_requested=false
python_pid=""

cleanup() {
    # Kill the Python process if it is running
    if [ -n "$python_pid" ]; then
        echo "Killing Python process..."
        kill "$python_pid"
        wait "$python_pid" 2>/dev/null
        python_pid=""
    fi
    kill_requested=false
}

trap 'cleanup' SIGINT

for ((number=10; number<=60; number++))
do
    file_path="kitchen_layouts_grid_text/kitchen${number}.txt"

    if [ "$kill_requested" = true ]; then
        cleanup
        continue
    fi

    # Run the Python command in the background and store the process ID
    # python 3d_plan_eval_main.py -m "vr" -k "$file_path" &
    python 3d_plan_eval_main.py -k "$file_path" &
    python_pid=$!

    # Continuously monitor the process status
    while true; do
        if ! ps -p "$python_pid" > /dev/null; then
            echo "Process has completed."
            kill_requested=true
            break
        fi

        # Check for user input to kill the process or continue
        if read -t 0.1 -n 1 -s input; then
            if [ "$input" = "X" ] || [ "$input" = "x" ]; then
                kill_requested=true
                cleanup
                break
            fi

            if [ "$input" = "C" ] || [ "$input" = "c" ]; then
                echo "Exiting..."
                exit 1
            fi
        fi
    done

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

# #!/bin/bash

# success_count=0
# total_files=0
# kill_requested=false
# python_pid=""

# cleanup() {
#     # Kill the Python process if it is running
#     if [ -n "$python_pid" ]; then
#         echo "Killing Python process..."
#         kill "$python_pid"
#         wait "$python_pid" 2>/dev/null
#         python_pid=""
#     fi
#     kill_requested=false
# }

# trap 'cleanup' SIGINT

# for ((number=3; number<=60; number++))
# do
#     file_path="kitchen_layouts_grid_text/kitchen${number}.txt"

#     if [ "$kill_requested" = true ]; then
#         cleanup
#         continue
#     fi

#     # Run the Python command in the background and store the process ID
#     python 3d_plan_eval_main.py -k "$file_path" &
#     python_pid=$!

#     # Wait for user input to kill the process
#     echo "Press X to kill the process or any other key to continue..."
#     read -n 1 -s input

#     if [ "$input" = "X" ] || [ "$input" = "x" ]; then
#         kill_requested=true
#         cleanup
#         continue
#     fi

#     if [ "$input" = "C" ] || [ "$input" = "c" ]; then
#         echo "Exiting..."
#         exit 1
#     fi

#     # Check if the process is still running
#     if ps -p "$python_pid" > /dev/null; then
#         echo "Process is still running..."
#         wait "$python_pid"
#     fi

#     log_file="test_logs/kitchen${number}_log.txt"

#     if [ -f "$log_file" ]; then
#         last_line=$(tail -n 1 "$log_file")

#         if [[ "$last_line" == "success" ]]; then
#             ((success_count++))
#         fi

#         ((total_files++))
#     fi
# done

# success_ratio=$(bc -l <<< "$success_count / $total_files")

# echo "Success ratio: $success_count / $total_files = $success_ratio"