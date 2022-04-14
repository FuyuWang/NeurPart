cd src/AdaptPart
CUDA_VISIBLE_DEVICES=0 python main.py --outdir outdir --fitness latency --cstr buffer_size --num_pe 65536 --l1_size 65536 --l2_size 25165824 --epochs 300 --df shi --num_steps 6 --test_epochs 10
cd ../../
