#python train.py --dataroot /home/xingzehang/project_18t/hxf/bubble_repair/training --name bubble_newcycle_add41bg --model cycle_gan  --batch_size 1 --display_port 8096 --lr 0.0002 --num_threads 4 --gpu_ids 0 --load_size 286 --crop_size 256 --display_winsize 256 --netG unet_256 --continue_train --epoch_count 62
#python train.py --dataroot /home/xingzehang/project_18t/hxf/bubble_repair/training --name bubble_new_add41bg_duibi --CUT_mode CUT --batch_size 1 --display_port 8096 --lr 0.0002 --num_threads 4 --gpu_ids 0 --load_size 286 --crop_size 256 --display_winsize 256
#python train.py --dataroot /home/xingzehang/project_18t/xzh/project/UNSB-main2023/datasets/fade2nofade_0315_add41bg --name fade2nofade_0315_add41bg --CUT_mode CUT --batch_size 1 --display_port 8098 --lr 0.0002 --num_threads 4 --gpu_ids 1 --load_size 286 --crop_size 256 --display_winsize 256 --continue_train --epoch_count 38
#没有identity 及sr loss
#python train.py --dataroot /home/xingzehang/project_18t/xzh/project/UNSB-main2023/datasets/fade2nofade_0315_add41bg --name fade2nofade_0315_add41bg_cyclegan --model cycle_gan --batch_size 1 --display_port 8098 --lr 0.0002 --num_threads 4 --gpu_ids 0 --load_size 286 --crop_size 256 --display_winsize 256 --netG unet_256 --continue_train --epoch_count 74
#没有sr loss
python train.py --dataroot /home/xingzehang/project_18t/xzh/project/UNSB-main2023/datasets/fade2nofade_0315_add41bg --name fade2nofade_0315_add41bg_cyclegan_noSR --model cycle_gan --batch_size 1 --display_port 8098 --lr 0.0002 --num_threads 4 --gpu_ids 0 --load_size 286 --crop_size 256 --display_winsize 256 --netG unet_256
