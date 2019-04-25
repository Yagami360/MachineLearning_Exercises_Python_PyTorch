```python
Start main()
PyTorch : 1.0.1.post2
----------------------------------------------
実行条件
----------------------------------------------
開始時間： 2019-04-25 02:13:28.970523
DEVICE :  GPU
NUM_EPOCHES :  10
LEARNING_RATE :  5e-05
BATCH_SIZE :  64
IMAGE_SIZE :  64
NUM_CHANNELS :  1
NUM_FEATURE_MAPS :  64
NUM_INPUT_NOIZE_Z :  100
NUM_CRITIC :  5
WEIGHT_CLAMP_LOWER :  -0.01
WEIGHT_CLAMP_UPPER :  0.01
実行デバイス : cuda
GPU名 : Tesla T4
torch.cuda.current_device() = 0
----------------------------------------------
ds_train : Dataset MNIST
    Number of datapoints: 60000
    Split: train
    Root Location: ./dataset
    Transforms (if any): Compose(
                             Resize(size=64, interpolation=PIL.Image.BILINEAR)
                             ToTensor()
                             Normalize(mean=(0.5,), std=(0.5,))
                         )
    Target Transforms (if any): None
ds_test : Dataset MNIST
    Number of datapoints: 10000
    Split: test
    Root Location: ./dataset
    Transforms (if any): Compose(
                             Resize(size=64, interpolation=PIL.Image.BILINEAR)
                             ToTensor()
                             Normalize(mean=(0.5,), std=(0.5,))
                         )
    Target Transforms (if any): None
dloader_train : <torch.utils.data.dataloader.DataLoader object at 0x7faf76896588>
dloader_test : <torch.utils.data.dataloader.DataLoader object at 0x7faf76896630>
Epoches:   0%|          | 0/10 [00:00<?, ?it/s]
minbatch process in DataLoader:   0%|          | 0/938 [00:00<?, ?it/s]----------------------------------
WassersteinGAN
<__main__.WassersteinGAN object at 0x7faf768967b8>
after init()
_device : cuda
_n_epoches : 10
_learning_rate : 5e-05
_batch_size : 64
_n_channels : 1
_n_fmaps : 64
_n_input_noize_z : 100
_n_critic : 5
_w_clamp_lower : -0.01
_w_clamp_upper : 0.01
_generator : Generator(
  (_layer): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace)
    (12): ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
_critic : Critic(
  (_layer): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  )
)
_loss_fn : None
_G_optimizer : Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.5, 0.999)
    eps: 1e-08
    lr: 5e-05
    weight_decay: 0
)
_C_optimizer : Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.5, 0.999)
    eps: 1e-08
    lr: 5e-05
    weight_decay: 0
)
----------------------------------
Starting Training Loop...

```