# Model=deit_small_patch16_224_shrink
# Model=deit_tiny_patch16_224_shrink
Model=deit_base_patch16_224_shrink
Done=[0,1,3,4,6,9]
CKPT=base_013469.pth
python main_lambda.py \
--data-set IMNET \
--data-path path/to/imagenet/ \
--model $Model \
--prune_layer [] \
--done_layer $Done \
--seed 42 \
--num_workers 10 \
--batch-size 256 \
--finetune $CKPT \
--eval \
--no-model-ema