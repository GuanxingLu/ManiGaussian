# this script is for evaluating a given checkpoint.
# example to evaluate our ManiGaussian:
#       bash scripts/eval.sh ManiGaussian_BC ${exp_name} 0
# Other examples:
#       bash scripts/eval.sh GNFACTOR_BC ${exp_name} 0
#       bash scripts/eval.sh PERACT_BC ${exp_name} 0

# some params specified by user
method_name=$1
exp_name=$2

# set the seed number
seed="0"
# set the gpu id for evaluation. we use one gpu for parallel evaluation.
eval_gpu=${3:-"0"}

cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"

use_split='test'    # or 'train' for debugging

starttime=`date +'%Y-%m-%d %H:%M:%S'`

if [ "${use_split}" == "train" ]; then
    echo "eval on train set"
    # eval on train set
    CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a python eval.py \
        method.name=$method \
        rlbench.task_name=${exp_name} \
        rlbench.demo_path=${train_demo_path} \
        framework.start_seed=${seed} \
        framework.eval_episodes=20

else
    echo "eval on test set"
    # eval on test set
    CUDA_VISIBLE_DEVICES=${eval_gpu} xvfb-run -a python eval.py \
        method.name=$method \
        rlbench.task_name=${exp_name} \
        rlbench.demo_path=${test_demo_path} \
        framework.start_seed=${seed}
fi

endtime=`date +'%Y-%m-%d %H:%M:%S'`
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);
echo "eclipsed time "$((end_seconds-start_seconds))"s"
