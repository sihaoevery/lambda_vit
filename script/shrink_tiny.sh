# PRUNE=[0,2,4]
PRUNE=[0,2,4,6]
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=14009 \
    --use_env main_lambda.py \
    --data-set IMNET \
    --data-path path/to/imagenet/ \
    --model deit_tiny_patch16_224_copy_lambda \
    --epochs 400 \
    --seed 42 \
    --prune_layer $PRUNE \
    --done_layer [] \
    --finetune deit_tiny_patch16_224-a1311bcf.pth \
    --batch-size 256 \
    --output_dir output