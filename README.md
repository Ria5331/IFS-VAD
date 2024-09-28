
# IFS-VAD

This is the official Pytorch implementation of our paper: "Inter-clip Feature Similarity based Weakly Supervised Video Anomaly Detection via Multi-scale Temporal MLP".




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



