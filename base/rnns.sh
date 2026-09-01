CUDA_VISIBLE_DEVICES=0 nohup uv run python run_experiments.py --config Reactive_High NoReplay RNN_Resid_1x8_seq Thorough2 &
CUDA_VISIBLE_DEVICES=1 nohup uv run python run_experiments.py --config Reactive_High NoReplay RNN_Resid_1x8 Thorough2 &
CUDA_VISIBLE_DEVICES=2 nohup uv run python run_experiments.py --config Reactive_High NoReplay RNN_Resid_2x4_seq Thorough2 &
CUDA_VISIBLE_DEVICES=3 nohup uv run python run_experiments.py --config Reactive_High NoReplay RNN_Resid_2x4 Thorough2 &
CUDA_VISIBLE_DEVICES=4 nohup uv run python run_experiments.py --config Reactive_High RNN_Resid_1x8_seq Thorough2 &
CUDA_VISIBLE_DEVICES=5 nohup uv run python run_experiments.py --config Reactive_High RNN_Resid_1x8 Thorough2 &
CUDA_VISIBLE_DEVICES=6 nohup uv run python run_experiments.py --config Reactive_High RNN_Resid_2x4_seq Thorough2 &
CUDA_VISIBLE_DEVICES=7 nohup uv run python run_experiments.py --config Reactive_High RNN_Resid_2x4 Thorough2 &