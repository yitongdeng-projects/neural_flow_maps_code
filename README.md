# [TOG/SIGGRAPH Asia 2023] Fluid Simulation on Neural Flow Maps
by [Yitong Deng](https://yitongdeng.github.io/), [Hong-Xing Yu](https://kovenyu.com/), Diyang Zhang, [Jiajun Wu](https://jiajunwu.com/), and [Bo Zhu](https://faculty.cc.gatech.edu/~bozhu/).

Our paper can be found at: [Coming Soon].

Video results can be found at: [Coming Soon].

## Environment
The environment can be installed by conda via:
```bash
conda env create -f environment.yml
conda activate nfm_env
```

Our code is tested on `Windows 11` with `CUDA 11.8`.

## Simulation

For running simulation, simply execute:
```bash
python run.py
```

Hyperparameters can be tuned by changing the values in the file `hyperparameters.py`. Checkpointing is available by setting the `from_frame` variable to the desired frame, given that the checkpoint of that frame can be found in `logs/[exp_name]/ckpts`.

## Visualization

The results will be stored in `logs/[exp_name]/vtks`. We recommend using ParaView to load these `.vti` files as a sequence and visualize them with volume.

## Bibliography
If you find our paper or code helpful, consider citing:
```
@article{deng2023neural,
title={Fluid Simulation on Neural Flow Maps},
author={Yitong Deng and Hong-Xing Yu and Diyang Zhang and Jiajun Wu and Bo Zhu},
journal={ACM Trans. Graph.},
volume={42},
number={6},
article={},
year={2023},
}
```
