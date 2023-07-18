


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 C 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 C 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 C 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 C 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 C 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 C 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 C 0 10 12 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 D 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 D 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 D 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 D 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 D 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 D 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 D 0 10 12 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-4 10 E 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-4 10 E 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-4 10 E 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 10 E 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 E 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 E 0 10 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 E 0 10 12 &
wait


<<comment

srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 C 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 C 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 C 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 C 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 C 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 C 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 C 0 5 12 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 D 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 D 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 D 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 D 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 D 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 D 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 D 0 5 12 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 E 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 E 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 E 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 E 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 E 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 E 0 5 12 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 E 0 5 12 &
wait



srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 C 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 C 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 C 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 C 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 C 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 C 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 C 0 5 2 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 D 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 D 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 D 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 D 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 D 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 D 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 D 0 5 2 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 E 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 E 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 E 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 E 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 E 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 E 0 5 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 E 0 5 2 &
wait





srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 C 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 C 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 C 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 C 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 C 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 C 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 C 0 5 8 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 D 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 D 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 D 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 D 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 D 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 D 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 D 0 5 8 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-7 5 E 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-6 5 E 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 5 E 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-4 5 E 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-3 5 E 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-2 5 E 0 5 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-1 5 E 0 5 8 &
wait




srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 C 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 C 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 C 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 C 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 C 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 C 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 C 0 10 2 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 D 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 D 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 D 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 D 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 D 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 D 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 D 0 10 2 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 E 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 E 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 E 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 E 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 E 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 E 0 10 2 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 E 0 10 2 &
wait



srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 C 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 C 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 C 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 C 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 C 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 C 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 C 0 10 8 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 D 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 D 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 D 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 D 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 D 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 D 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 D 0 10 8 &
wait


srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-5 10 E 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-5 10 E 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-5 10 E 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 1e-5 10 E 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 9e-6 10 E 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 6e-6 10 E 0 10 8 &
srun --gres=gpu:1 -C cuda-mode-exclusive -t 360 -N 1 -n 1 python new-GATv2-4-layers-no-hist.py tech-earnings-calls daily_price_change no-hist 3e-6 10 E 0 10 8 &
wait
comment