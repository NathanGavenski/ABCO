# Augmented Behavior Cloning from Observation (ABCO)

Official Pytorch implementation of [Augmented Behavior Cloning from Observation](https://arxiv.org/abs/2004.13529)

## Downloading the data
You can download the data we used to train our models [here](https://drive.google.com/file/d/1_wnrfv1OEM_EuPaF5tMF2l2ZJjr9lJVh/view?usp=sharing)

## Training ABCO

After downloading the expert demonstration, you can then train ABCO. There are several training scripts in the directory. 

```
./scripts/bcio_alpha_3
./scripts/bcio_alpha_5
./scripts/bcio_alpha_10
./scripts/bcio_alpha_acrobot
./scripts/bcio_alpha_cartpole
./scripts/bcio_alpha_mountaincar
```
**We ran ABCO on a server, if you are running locally you might want to remove** ```xvfb-run -a -s "-screen 0 1400x900x24"``` **from the scripts**

## Citation

```
@article{monteiro2020augmented,
  title={Augmented Behavioral Cloning from Observation},
  author={Monteiro, Juarez and Gavenski, Nathan and Granada, Roger and Meneguzzi, Felipe and Barros, Rodrigo},
  journal={arXiv preprint arXiv:2004.13529},
  year={2020}
}
```

### ToDo
- [x] requirements
- [ ] finish the ReadMe
