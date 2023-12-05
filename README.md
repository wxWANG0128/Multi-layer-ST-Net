# Multi-layer-ST-Net
This is a multi-layer spatial-temporal network for traffic prediction. This project was done as the assignment for the course CE634.

# Requirements
Please note that the package `TransBigData` usually requires extra packages. Please see [Guide](https://transbigdata.readthedocs.io/en/latest/#).
```
pip install -r requirements.txt
```
# Data Preparation
The data structure in this model obeys the structure used in `DCRNN`[(Li et al., 2018)](https://github.com/liyaguang/DCRNN). The `metr-la` and `PEMS-BAY` dataset was provided by [Li et al. (2018)](https://github.com/liyaguang/DCRNN). The `PEMS-BAY23` dataset was collected from [Caltrans](https://pems.dot.ca.gov/). This dataset contains data observed from 325 sensors in bay area from 01/01/2023 to 30/08/2023. These two datasets can be downloaded from:

[metr-la](https://drive.google.com/file/d/1B3ue_5JOiirEYpLC12pMpitZ09jEvB6g/view?usp=drive_link)

[PEMS-BAY](https://drive.google.com/file/d/1Z6FLyWyPIcJeT9T_sSUKf3zsaWlMb1Gj/view?usp=sharing)

[PEMS-BAY23](https://drive.google.com/file/d/1tXeodobgp3n9CnoSQdmPKWxQhs4h5Mq4/view?usp=drive_link)

To prepare the data for training, Please run:
```
# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/BAY-NEW --traffic_df_filename=data/pems-bay_0.h5
```
