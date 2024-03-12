# example to run our ManiGaussian:
#       bash scripts/train_and_eval_w_geo_sem.sh ManiGaussian_BC 0,1 12345 ${exp_name}
# Other examples:
#       bash scripts/train_and_eval_w_geo_sem.sh GNFACTOR_BC 0,1 12345 ${exp_name}

# set the method name
method=${1}

# set the seed number
seed="0"
# set the gpu id for training. we use two gpus for training. you could also use one gpu.
train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=True

cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"

# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}
replay_dir="${cur_dir}/replay/${exp_name}"

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 5s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}

#######
# override hyper-params in config.yaml
#######
batch_size=1
tasks=[close_jar,open_drawer,sweep_to_dustpan_of_size,meat_off_grill,turn_tap,slide_block_to_color_target,put_item_in_drawer,reach_and_drag,push_buttons,stack_blocks]
demo=20
lambda_embed=0.01   # default: 0.01
render_freq=2000

tmux select-pane -t 0 
tmux send-keys "conda activate manigaussian; CUDA_VISIBLE_DEVICES=${train_gpu} python train.py method=$method \
rlbench.task_name=${exp_name} \
rlbench.demo_path=${train_demo_path} \
replay.path=${replay_dir} \
framework.start_seed=${seed} \
framework.use_wandb=${use_wandb} \
method.use_wandb=${use_wandb} \
framework.wandb_group=${exp_name} \
framework.wandb_name=${exp_name} \
ddp.num_devices=${#train_gpu_list[@]} \
replay.batch_size=${batch_size} \
ddp.master_port=${port} \
rlbench.tasks=${tasks} \
rlbench.demos=${demo} \
method.neural_renderer.render_freq=${render_freq} \
method.neural_renderer.lambda_embed=${lambda_embed} \
method.neural_renderer.foundation_model_name=diffusion" C-m

# remove 0.ckpt
rm -rf logs/${exp_name}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}
