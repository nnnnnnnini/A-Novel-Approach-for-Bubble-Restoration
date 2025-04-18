<<<<<<< HEAD
# A-Novel-Approach-for-Bubble-Restoration
We propose a novel method for bubble restoration in pathological images. This method combines CycleGAN with contrastive learning to better restore cell-level details within the bubbles.

The code is in the Master branch.
=======


# A Novel Approach for Bubble Restoration in Histopathological Images

We propose a novel method for bubble restoration in pathological images. This method combines CycleGAN with contrastive learning to better restore cell-level details within the bubbles.

### Training and Test

The data used for training are expected to be organized as follows:
```bash
Data_Path                # DIR_TO_TRAIN_DATASET
 ├──  trainA
 |      ├── 1.png     
 |      ├── ...
 |      └── n.png
 ├──  trainB     
 |      ├── 1.png     
 |      ├── ...
 |      └── m.png
 ├──  testA
 |      ├── 1.png     
 |      ├── ...
 |      └── j.png
 └──  testB     
        ├── 1.png     
        ├── ...
        └── k.png

```

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

- Train the model:
```bash
python train.py --dataroot /home/bubble_repair/training{dataroot_train_dir_name} --name ${model_results_dir_name} --model cycle_gan  --batch_size 1 --display_port 8096 --lr 0.0002 --num_threads 4 --gpu_ids 0 --load_size 286 --crop_size 256 --display_winsize 256 --netG unet_256 
```

- Test the model:
```bash
python test.py --dataroot /home/bubble_repair/training{dataroot_train_dir_name} --name ${model_results_dir_name} --model cycle_gan --phase test  --epoch ${epoch_number}  --num_test ${number_of_test_images} --results_dir ${result_dir_name} --netG unet_256 --gpu_ids 0 --load_size 256 --crop_size 256 --display_winsize 256
```

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:



### Acknowledgments
Our code is developed based on [FFPE++](https://github.com/DeepMIALab/FFPEPlus).
>>>>>>> copymaster
