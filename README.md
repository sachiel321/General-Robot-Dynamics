# General Robot Dynamics (GRD)

## This is the official implementation code of paper "General Robot Dynamics Learning and Gen2Real"

Acquiring dynamics is an essential topic in robot
learning, but up-to-date methods, such as dynamics randomization, need to restart to check nominal parameters, generate
simulation data, and train networks whenever they face different
robots. To improve it, we novelly investigate general robot
dynamics, its inverse models, and Gen2Real, which means transferring to
reality. Our motivations are to build a model that learns the
intrinsic dynamics of various robots and lower the threshold of
dynamics learning by enabling an amateur to obtain robot models
without being trapped in details. This paper achieves the “generality”
by randomizing dynamics parameters, topology configurations,
and model dimensions, which in sequence cover the property, the
connection, and the number of robot links. A structure modified
from GPT is applied to access the pre-training model of general
dynamics. We also study various inverse models of dynamics to
facilitate different applications. We step further to investigate a
new concept, “Gen2Real”, to transfer simulated, general models
to physical, specific robots.

We implemented the GRD model by PyTorch. You can find our trained model from [TrainedModel](https://drive.google.com/drive/folders/1BO6OSiVth9fmsIcNPjRarbaYIaGnPP_V?usp=sharing), which can be load by file `GRD.py`.

- `GRD.py` is the entry program, in which you can change the model configs.
- `core/model.py` contains the actual General model (the GPT part borrows from [Karpathy's](https://github.com/karpathy/minGPT) implementation)
- `core/trainer.py` trains and test the model.
- `data/my_robot.m` You can generate robot simulation data by this file. By modifying the file you can customize the random range of the robot.

Supported Platforms:

- Ubuntu
- CentOS

### Example usage
```python

# You can run GRD-Dynamic Gen2Real demo from the Linux command line as follows
CUDA_VISIBLE_DEVICES=0 python  -m torch.distributed.launch --nproc_per_node=1 GRD.py

# Reverse Dynamic training
CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch --nproc_per_node=2 GRD.py --mode_select=train --model_select=Reverse_Dynamic --batch_size=10 --learning_rate=2e-4 --learning_decay=True --train_epochs=1000  
```

### Citation

Please use the following bibtex entry:
```
@article{Xing2021generative,
  title={General Robot Dynamics Learning and Gen2Real},
  author={Xing,},
  year={2021}
}
```

### License

MIT
