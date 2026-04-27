#!/bin/bash
# Monitor training progress

TASK_OUTPUT="/tmp/claude-1000/-home-glcuser-projhighcv-bone-tumor/tasks/b70de0d.output"

echo "=== Monitoring Training Progress ==="
echo "Task output file: $TASK_OUTPUT"
echo ""

while true; do
    clear
    echo "=== Training Progress (Last 50 lines) ==="
    echo "Time: $(date)"
    echo ""

    if [ -f "$TASK_OUTPUT" ]; then
        tail -50 "$TASK_OUTPUT" | grep -E "Epoch=|val_dice=|Best Dice|Training completed|OOM"
    else
        echo "Waiting for training to start..."
    fi

    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

    sleep 30
done
