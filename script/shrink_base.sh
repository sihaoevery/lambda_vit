# PRUNE=[0,1,3,4,6]
PRUNE=[0,1,3,4,6,9]
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=14009 \
    --use_env main_lambda.py \
    --data-set IMNET \
    --data-path path/to/imagenet/ \
    --model deit_base_patch16_224_attn \
    --epochs 400 \
    --seed 42 \
    --prune_layer $PRUNE \
    --done_layer [] \
    --finetune deit_base_patch16_224-b5f2ef4d.pth \
    --batch-size 256 \
    --output_dir output