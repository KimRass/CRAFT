- https://github.com/HCIILAB/Scene-Text-Detection#5-ocr-service

- Region score map predicted using pre-trained model
    - <img src="https://i.imgur.com/tSjlj5b.jpg" width="800">
- Region score map generated using annotated bounding boxes
    - <img src="https://i.imgur.com/Lj0r973.jpg" width="800">


# Online Hard Example Mining
- [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/pdf/1604.03540.pdf)
- Reference: https://sh-tsang.medium.com/review-ohem-training-region-based-object-detectors-with-online-hard-example-mining-object-ad791ad87612
- It is assumed that regions with some overlap with the ground truth are more likely to be the confusing or hard ones.
- Although this heuristic helps convergence and detection accuracy, it is suboptimal because it ignores some infrequent, but important, difficult background regions.
- **To handle the data imbalance, heuristics is designed to rebalance the foreground-to-background ratio in each mini-batch to a target of $1 : 3$ by undersampling the background patches at random, thus ensuring that 25% of a mini-batch is fg RoIs.**
- The loss of each RoI represents how well the current network performs on each RoI.
- **Hard examples are selected by sorting the input RoIs by loss and taking the B/N examples for which the current network performs worst.**
- And OHEM does not need a fg-bg ratio for data balancing. If any class were neglected, its loss would increase.
- There can be images where the fg RoIs are easy (e.g. canonical view of a car), so the network is free to use only bg regions in a mini-batch; and vice versa when bg is trivial (e.g. sky, grass etc.), the mini-batch can be entirely fg regions.
- The implementation maintains two copies of the RoI network, one of which is readonly.
- The readonly RoI network performs a forward pass and computes loss for all input RoIs (R) (green arrows).
- Then the hard RoI sampling module uses OHEM to select hard examples (Rhard-sel), which are input to the regular RoI network (red arrows).
This network computes forward and backward passes only for Rhard-sel.

- Object detectors are often trained through a reduction that converts object detection into an image classification problem. *This reduction introduces a new challenge that is not found in natural image classification tasks: the training set is distinguished by a large imbalance between the number of annotated objects and the number of background examples (image regions not belonging to any object class of interest). In the case of sliding-window object detectors this imbalance may be as extreme as 100,000 background examples to every one object.*
- Our motivation is the same as it has always been – detection datasets contain an overwhelming number of easy examples and a small number of hard examples. Automatic selection of these hard examples can make training more effective and efficient. OHEM is a simple and intuitive algorithm that eliminates several heuristics and hyperparameters in common use.
- To handle the data imbalance, designed heuristics to rebalance the foreground-to-background ratio in each mini-batch to a target of $1 : 3$ by undersampling the background patches at random, thus ensuring that 25% of a mini-batch is $fg$ RoIs.
