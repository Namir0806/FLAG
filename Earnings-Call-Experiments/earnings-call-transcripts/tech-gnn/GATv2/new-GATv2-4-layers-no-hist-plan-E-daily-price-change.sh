<<comment
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 E 0 5 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 E 0 5 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 E 0 5 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 E 0 5 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 E 0 5 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 E 0 5 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 E 0 5 4 &
wait


comment

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 E 0 10 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 E 0 10 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 E 0 10 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 E 0 10 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 E 0 10 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 E 0 10 4 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 E 0 10 4 &
wait

