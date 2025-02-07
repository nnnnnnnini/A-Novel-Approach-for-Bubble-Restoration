

# A Novel Approach for Bubble Restoration in Histopathological Images

In this work, we introduce FFPE++ to improve the quality of FFPE tissue sections using an unpaired image-to-image translation technique that converts FFPE images with artifacts into high-quality FFPE images without the need for explicit image pairing and annotation.

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

The test results will be saved to a html file here: ``` ./results/${result_dir_name}/latest_train/index.html ``` 

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:



### Acknowledgments
Our code is developed based on [FFPEPlus](https://github.com/DeepMIALab/FFPEPlus).
