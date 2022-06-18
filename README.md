# Joint Synthesis of Safety Certificate and Safe Control Policy Using Constrained Reinforcement Learning

This repository is the official implementation of [*Joint Synthesis of Safety Certificate and Safe Control Policy Using Constrained Reinforcement Learning*](https://proceedings.mlr.press/v168/ma22a.html).
The code base of this implementation is the [Parallel Asynchronous Buffer-Actor-Learner (PABAL) architecture](https://github.com/idthanm/mpg),
which includes implementations of most common RL algorithms with the state-of-the-art training efficiency.
If you are interested in or want to contribute to PABAL, you can contact me or the [original creator](https://github.com/idthanm). I also [reimplemented it with ppo on TF1](https://github.com/mahaitongdae/Safety_Index_Synthesis/tree/tf1) after the paper was submitted. TF1 code is directly modified from [safety-gym open-sourced code](https://github.com/openai/safety-starter-agents), which is easier to setup and run. The results of two versions only have differences in the early training stage, which does not affect the claimed performance.

## Requirements
First, install [Safety-gym](https://github.com/openai/safety-gym). Then you might replace the engine.py in safety-gym package with our [custom engine](https://github.com/mahaitongdae/Safety_Index_Synthesis/blob/master/utils/engine_custom.py). We modify the engine.py to estimate the distance and velocity, so that the code could be more general to any other tasks with the distance and velocity observations/signals.

To install other requirements:

```setup up your anaconda env or virtualenv in python 3.6 (Higher version may be mot compatible with safety-gym)
$ pip install -U ray
$ pip install tensorflow==2.5.0
$ pip install tensorflow_probability==0.13.0
$ pip install seaborn matplotlib
```

## Training
To train the algorithm(s) in the paper, run these commands or directly run `sh bash.sh` in `/train_scripts/`:
```train
$ export PYTHONPATH=/your/path/to/Reachability_Constrained_RL/:$PYTHONPATH
$ cd ./train_scripts/
$ python train_scripts4fsac.py                # FAC-SIS / FAC-\phi_0,\phi_h (changing if updating \phi and the init \phi in config)
```


### Training supervision
Results can be seen with tensorboard:
```
$ cd ./results/
$ tensorboard --logdir=. --bindall
```

## Evaluation
To test and evaluate trained policies, run:

```test
python train_scripts4fsac.py --mode testing --test_dir <your_log_dir> --test_iter_list <iter_nums>
```
and the results will be recored in `/results/<ENV_NAME>/<ALGO_NAME>/<EXP_TIME>/logs/tester`.

## Contributing
When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with me before making a change.

## Feel free to cite our paper with BibTex:
```
@InProceedings{ma22joint,
  title = 	 {Joint Synthesis of Safety Certificate and Safe Control Policy Using Constrained Reinforcement Learning},
  author =       {Ma, Haitong and Liu, Changliu and Li, Shengbo Eben and Zheng, Sifa and Chen, Jianyu},
  booktitle = 	 {Proceedings of The 4th Annual Learning for Dynamics and Control Conference},
  pages = 	 {97--109},
  year = 	 {2022},
  editor = 	 {Firoozi, Roya and Mehr, Negar and Yel, Esen and Antonova, Rika and Bohg, Jeannette and Schwager, Mac and Kochenderfer, Mykel},
  volume = 	 {168},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--24 Jun},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v168/ma22a/ma22a.pdf},
  url = 	 {https://proceedings.mlr.press/v168/ma22a.html},
}

```
## Other related papers if interested:
* [Reachability Constrained Reinforcement Learning](https://arxiv.org/abs/2205.07536)
* [Feasible Actor-Critic: Constrained Reinforcement Learning for Ensuring Statewise Safety](https://arxiv.org/abs/2105.10682)
* [Learn Zero-Constraint-Violation Policy in Model-Free Constrained Reinforcement Learning](https://arxiv.org/abs/2111.12953)
