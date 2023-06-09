#!/bin/bash

success_count=0
total_files=0

for ((number=3; number<=60; number++))
do
    file_path="kitchen_layouts_grid_text/kitchen${number}.txt"
    
    python 3d_plan_eval_main.py -k "$file_path"
    
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