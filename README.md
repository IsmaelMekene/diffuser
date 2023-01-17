# Planning with Diffusion &nbsp;&nbsp; 
This project is built on the top of the Diffuser [paper [1]](https://arxiv.org/abs/2205.09991) and its [code](https://github.com/jannerm/diffuser/tree/main).

The original [Diffuser](https://diffusion-planning.github.io/) model developped by generates trajectories by iteratively denoising randomly sampled plans. The Diffusion is performed on the sequences of states-actions pairs $[s_t,a_t,...,s_T,a_T]$ (*T* denotes the horizon)

Our first extension is to reduce training and inference time by adapting the model such that  diffusion is only performed on the sequence of future actions $[a_t,...,a_T]$ (to enable the model to handle efficently *image based observations*, and not only state-vector observations). Crucially, the current state s_t is given to the model as input, like in conditional diffusion, but diffusion is only performed on the sequences of action. Thus, the new input is [a_t,...,a_T,s_t] instead of [a_t,...,a_T,s_t]. (expand to see more details)



## Quickstart
Clone this repo.

## Installation

1. Install mujoco and set the key. For example, one may use these commands

```
mkdir ~/.mujoco
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz 
wget https://www.roboti.us/file/mjkey.txt
cp mjkey.txt ~/.mujoco/mujoco210/bin
```

2. Install dependencies and an virtual environment
```
conda env create -f environment.yml
conda activate diffuser
pip install -e .
```

## Using pretrained models
We train (several versions)  Diffuser with the extensions described above in the mujoco in the mujoco environments *halfcheetah* and *walker-2d* (using the D4RL medium-expert dataset). One can download the pretrained weights from this drive [link](https://drive.google.com/drive/folders/11tfJ8XO1pYn3gEhs5Wg14M2i9k2VYw-m?usp=sharing)

### Downloading weights from the original paper

Download pretrained diffusion models and value functions of the original [paper](https://arxiv.org/abs/2205.09991)  with:
```
./scripts/download_pretrained.sh
```

This command downloads and extracts a [tarfile](https://drive.google.com/file/d/1srTq0OFQtWIv9A7fwm3fwh1StA__qr6y/view?usp=sharing) containing [this directory](https://drive.google.com/drive/folders/1ie6z3toz9OjcarJuwjQwXXzDwh1XnS02?usp=sharing) to `logs/pretrained`. The models are organized according to the following structure:
```
└── logs/pretrained
    ├── ${environment_1}
    │   ├── diffusion
    │   │   └── ${experiment_name}
    │   │       ├── state_${epoch}.pt
    │   │       ├── sample-${epoch}-*.png
    │   │       └── {dataset, diffusion, model, render, trainer}_config.pkl
    │   └── values
    │       └── ${experiment_name}
    │           ├── state_${epoch}.pt
    │           └── {dataset, diffusion, model, render, trainer}_config.pkl
    ├── ${environment_2}
    │   └── ...
```

The `state_${epoch}.pt` files contain the network weights and the `config.pkl` files contain the instantation arguments for the relevant classes.
The png files contain samples from different points during training of the diffusion model.

### Planning

### Unguided planning (see description above)

To generate a sequence of future actions [a_t,...,a_{t+horizon}] up to an *horizon* (below we use horizon=32), run

```
python scripts/plan_unguided.py --horizon 32 --dataset walker2d-medium-expert-v2 --logbase /logs
```

## Training from scratch

1. Train **an unguided** diffusion model with:
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```

The default hyperparameters are listed in [locomotion:diffusion](config/locomotion.py#L22-L65).
You can override any of them with flags, eg, `--n_diffusion_steps 100`.


2. Plan using your newly-trained models with the same command as in the pretrained planning section, simply replacing the logbase to point to your new models:
```
python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs
```
See [locomotion:plans](config/locomotion.py#L110-L149) for the corresponding default hyperparameters.

**Deferred f-strings.** Note that some planning script arguments, such as `--n_diffusion_steps` or `--discount`,
do not actually change any logic during planning, but simply load a different model using a deferred f-string.
For example, the following flags:
```
---horizon 32 --n_diffusion_steps 20 --discount 0.997
--value_loadpath 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}'
```
will resolve to a value checkpoint path of `values/defaults_H32_T20_d0.997`. It is possible to
change the horizon of the diffusion model after training (see [here](https://colab.research.google.com/drive/1YajKhu-CUIGBJeQPehjVPJcK_b38a8Nc?usp=sharing) for an example),
but not for the value function.


## Reference
```
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```


## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.
