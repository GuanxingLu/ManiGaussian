# this script generates demonstrations for multiple tasks sequentially.
# example:
#       bash scripts/gen_demonstrations_all.sh

# The recommended 10 tasks
ALL_TASK="close_jar open_drawer sweep_to_dustpan_of_size meat_off_grill turn_tap slide_block_to_color_target put_item_in_drawer reach_and_drag push_buttons stack_blocks"

tasks=($ALL_TASK)

for task in "${tasks[@]}"; do
    echo "###Generating demonstrations for task: $task"
    bash scripts/gen_demonstrations.sh "$task"
done
