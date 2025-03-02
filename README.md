# IFS-VAD

This is the official Pytorch implementation of our paper: "[**Inter-clip Feature Similarity based Weakly Supervised Video Anomaly Detection via Multi-scale Temporal MLP**](https://ieeexplore.ieee.org/document/10720820) " in TCSVT 2025.




## Setup

We use the extracted CLIP and I3D features for ShanghaiTech, UCF-Crime and XD-Violence datasets from the following works:

[CLIP features for ShanghaiTech, UCF-Crime and XD-Violence](https://github.com/joos2010kj/CLIP-TSA) 

[I3D features for ShanghaiTech](https://github.com/tianyu0207/RTFM)  

[I3D features for UCF-Crime](https://github.com/Roc-Ng/DeepMIL)  

[I3D features for XD-Violence](https://roc-ng.github.io/XD-Violence/)

The following files need to be adapted in order to run the code on your own machine:

* Change the file paths to the download datasets above in `ucf-clip-test-10crop.list` and `ucf-clip-train-10crop.list`.

* Feel free to change the hyperparameters in `option.py`
## Train and test

After the setup, simply run the following commands:

    #train
    python main.py --feature_extractor [clip/i3d] --dataset [shanghai/ucf/xd] --gpus 0
    
    #test
    python test_main.py --feature_extractor [clip/i3d] --dataset [shanghai/ucf/xd] --gpus 0




## Citation

If you find this repo helpful for your research, please consider citing our paper:

    @ARTICLE{Zhu2025IFS,
        author={Zhong, Yuanhong and Zhu, Ruyue and Yan, Ge and Gan, Ping and Shen, Xuerui and Zhu, Dong},
        journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
        title={Inter-Clip Feature Similarity Based Weakly Supervised Video Anomaly Detection via Multi-Scale Temporal MLP}, 
        year={2025},
        volume={35},
        number={2},
        pages={1961-1970},
        keywords={Feature extraction;Training;Anomaly detection;Annotations;Circuits and systems;Multilayer perceptrons;Surveillance;Germanium;Explosions;Transformers;Video anomaly detection;weakly supervised learning;multiple instance learning;multilayer perceptron},
        doi={10.1109/TCSVT.2024.3482414}}
