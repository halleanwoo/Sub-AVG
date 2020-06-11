# Sub-AVG

This repo is the source code of the paper "Sub-AVG: Overestimation Reduction for Cooperative Multi-Agent Reinforcement Learning". It is constructed based on the [SMAC](https://github.com/oxwhirl/smac) and [PyMARL](https://github.com/oxwhirl/pymarl).

## Installation

Clone this repo and set up SMAC:

```sh
cd Sub_AVG
bash install_sc2.sh
```

The installation is same with [PyMARL](https://github.com/oxwhirl/pymarl). For quick start, you can install the necessary packages directly by (not recommended):

```sh
pip install -r requirements.txt
```

If needed, add  directory that contains StarCraftII to the system's path ('xxx' is your local file path):

```sh
export PATH="$PATH:/xxx/Sub_AVG/3rdparty/StarCraftII"
```

## Run Sub-AVG

We provide  `run_QMIX_vanilla.sh` , `run_QMIX_Double.sh` and `run_QMIX_Sub_AVG_Mixer.sh` for quick start.

The parameters used in the shell is summarized as follows:

- `double_q`

  Whether to use the Double Agent Network;

- `SubAVG_Agent_flag`

  Whether to use the Sub-AVG Agent;

- `SubAVG_Agent_K`

  How many *target agent networks* are used;

- `SubAVG_Agent_flag_select`

  - -1 :   Sub-AVG Agent;   
  - 0  :   Averaged Agent;
  - 1  ：Over-AVG Agent;

- `SubAVG_Mixer_flag`

  Whether to use the Sub-AVG Mixer;

- `SubAVG_Mixer_K`

  How many *target mixing networks* are used;

- `SubAVG_Mixer_flag_select`

  - -1 :   Sub-AVG Mixer;   
  - 0  :   Averaged Mixer;
  - 1  ：Over-AVG Mixer;

- `env_args.map_name`

  Change the SMAC scenario;

- `--config=`

  - qmix_smac:  run QMIX;
  - vdn_smac  :  run  VDN;
  - smix              :  run SMIX($\lambda$);

- Notice

  When `SubAVG_Agent_flag` and `SubAVG_Mixer_flag` are both True(i.e. =1), it is running *Sub-AVG Both*.

## Results

The results save path is:  "./results/sacred".

The experimental results are presented in Table 1:

<img src='http://wl.lelego.top/Table1.png' style='width:700px'>

