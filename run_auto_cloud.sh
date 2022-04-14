cd src/AutoPart2
CUDA_VISIBLE_DEVICES=0 python main.py --outdir outdir --fitness latency --cstr buffer_size --num_pe 65536 --l1_size 65536 --l2_size 25165824 --epochs 500 --df shi
cd ../../