*This is the official code for the paper **ExAgt: Expert-guided Augmentation for Representation Learning of Traffic Scenarios** published in IEEE ITSC 2022, Macau, China*


**Dataset**

* The Argoverse motion prediction dataset is pre-processed and the augmentation and occcupancy grids information are generated. Please download the dataset from the link <will be added shortly>
* Place the dataset in a folder named data 

**Code**
* The file `main.py` is the config file to start the experiment
* The main config in the file is the loss_type to be used and the eval metric. The user can change these two values depending on the task under consideration https://github.com/lab176344/ExAgt_Work/blob/d758e4b45fb82b7a4a6ae96fec0a0655a9cd97c2/main.py#L82 and https://github.com/lab176344/ExAgt_Work/blob/d758e4b45fb82b7a4a6ae96fec0a0655a9cd97c2/main.py#L83
* The main file will also start the tensorboard visualisation of the training
* The setting parameters for the training and augmentation are explained in the paper, the user can reproduce the settings


Please cite our paper if you find the code useful
```
@article{balasubramanian2022exagt,
  title={ExAgt: Expert-guided Augmentation for Representation Learning of Traffic Scenarios},
  author={Balasubramanian, Lakshman and Wurst, Jonas and Egolf, Robin and Botsch, Michael and Utschick, Wolfgang and Deng, Ke},
  journal={arXiv preprint arXiv:2207.08609},
  year={2022}
}
```

If you have any issues with the code or any doubts raise an issue in the repo, we will try to resolve it as soon as possible

