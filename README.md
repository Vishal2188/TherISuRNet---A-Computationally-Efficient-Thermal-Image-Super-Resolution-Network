# TherISuRNet - A Computationally Efficient Thermal Image Super-Resolution Network 

Download the
[Paper - CVPRW](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w6/Chudasama_TherISuRNet_-_A_Computationally_Efficient_Thermal_Image_Super-Resolution_Network_CVPRW_2020_paper.pdf) 

## Network design of the proposed TherISuRNet model:
<img src = "network design/ThermalSR.png" width = 800>

## Residual network (ResBlock):
<img src = "network design/ResBlock.png" width = 600>

## Usage 

1. Download the [PBVS Training dataset.](https://drive.google.com/file/d/17kZOAQqRBVH7o4S87RtD90-P51iWeDFV/view?usp=sharing)

* Train Data bicubic downscaling x4 (LR images)
* Train Data (HR images)
* Validation Data bicubic downscaling x4 (LR images)
* Validation Data (HR images)

  Put the downloaded dataset in the train_lr, train_hr, val_lr and val_hr folders.
  like this
```
...
│
├── train_lr
│     ├── 1.png
│     ├── 2.png
│     ├── ...
├── train_hr
│     ├── 1.png
│     ├── 2.png
│     ├── ...
├── val_lr
│     ├── 1.png
│     ├── 2.png
│     ├── ...
├── val_hr
│     ├── 1.png
│     ├── 2.png
│     ├── ...
├── main.py
├── model.py
...
```

2. Download the [imagenet-vgg-verydeep-19.mat](https://drive.google.com/file/d/1tMZ59K4wTunDnkxNBREXR_c00hyMwyWY/view?usp=sharing) file.
 
(Note - Only if you want to use Contextual loss [3] in training process)


3. Train TherISuRNet model.

```
python main.py
```

4. After training, inference can be performed.

Download the testing LR and HR images from here 

[PBVS Validation dataset :](https://drive.google.com/file/d/1-0zhyiseHB5B9ha41zIHggUpUJ5BAHJx/view?usp=sharing) 50 number of images [1]

[FLIR Validation dataset :](https://drive.google.com/file/d/1urP-f3EhehwqY-kjSQkVc0qWgTwd9fzC/view?usp=sharing) 1366 number of images

[KAIST Validation dataset :](https://drive.google.com/file/d/1QPBnbjbLIubw_4xni1YgqoauvZ1_v4Gx/view?usp=sharing) 500 preprocessed images

Put above test images and/or your own test images that you want to evaluate into a test folder

example:
 ```
 ...
│
├── main.py
├── pred.py
├── model
│     ├── checkpoint
│     ├── model.ckpt-100000
│     ├── model.ckpt-100000
│     └── model.ckpt-100000
├── test
│     ├── yourpic1.png
│     ├── favpic.jpg
│     ...
│     └── smallpic.png
...
```
 and run the following command.
```
python pred.py test
```

5. To calculate the corresponding PSNR and SSIM measures

```
python PSNR_SSIM.py
```

## Result examples

The SR results of the TherISuRNet model can be downloaded from [here.](https://drive.google.com/file/d/1JbkX842TsWZcbxecu7Wr7rVOo-C8Rd2K/view?usp=sharing)

### Computationally Efficiency Comparison:
<img src = "network design/SSIMParameters.png" width = 900>

### Quantitative Comparison:
<img src = "network design/QuantitativeComparison.png" width = 700>

## References
1) Rafael E. Rivadeneira, Angel D. Sappa, and Boris X. Vintimilla. Thermal image super-resolution: a novel architecture and dataset. In International Conference on Computer Vision Theory and Applications, pages 1–2, 2020.
2) Purbaditya Bhattacharya, Jorg Riechen, and Udo Zolzer. Infrared image enhancement in maritime environment with convolutional neural networks. In VISIGRAPP, 2018.
3) Mechrez, Roey and Talmi, Itamar and Zelnik-Manor, Lihi, The Contextual Loss for Image Transformation with Non-Aligned Data, arXiv preprint arXiv:1803.02077, 2018.

## If you use this repositary, then please cite our paper in your publications as
V. Chudasama et al., "TherISuRNet - A Computationally Efficient Thermal Image Super-Resolution Network," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Seattle, WA, USA, 2020, pp. 388-397, doi: 10.1109/CVPRW50498.2020.00051.

**Bibtex file:**

@INPROCEEDINGS{TherISuRNet,  
     author={V. {Chudasama} and H. {Patel} and K. {Prajapati} and K. {Upla} and R. {Ramachandra} and K. {Raja} and C. {Busch}},  
     booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},   
     title={TherISuRNet - A Computationally Efficient Thermal Image Super-Resolution Network},   
     year={2020},  
     pages={388-397},  
     doi={10.1109/CVPRW50498.2020.00051}}
