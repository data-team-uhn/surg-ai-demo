python3 -m sergey_code.new.train_valid_multi_task /cluster/projects/madanigroup/CLAIM/sergey/jobs/train_valid_multi_task-11861460-11 data dangerous_safe --task_loss_weights uniform --train_batch_size 128 --val_test_batch_size 128 --epochs 400 --initial_lr 1e-6 --max_lr 1e-2 --weight_decay 1e-2 --segformer_model_variant MiT-b0