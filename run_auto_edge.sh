cd src/AutoPart
CUDA_VISIBLE_DEVICES=1 python main.py --outdir outdir --fitness latency --cstr buffer_size --num_pe 4096 --l1_size 4096 --l2_size 25165824 --epochs 500 --df dla
cd ../../