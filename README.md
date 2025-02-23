# NPN: Adaptive Integration of Partial Label Learning and Negative Learning for Enhanced Noisy Label Learning
**Abstract:** Traditional label-noise learning (LNL) paradigms (e.g., sample selection, label correction, and sample re-weighting) have achieved significant success in combating noisy labels.  However, they heavily rely on extensive prior knowledge (e.g., noise rates) to ensure performance, which is often unavailable in large-scale, real-world benchmarks. To address these limitations, we propose a novel LNL paradigm, termed CA2C, which eliminates reliance on prior knowledge by integrating a Combined Asymmetric Co-learning and Co-training framework. To do this, CA2C first introduces an asymmetric co-learning strategy with paradigm decomposition to maximize the distinct learning capabilities of different paradigms, utilizing distinct learning paradigms trained on twin models rather than a single model with a joint loss. Additionally, to mitigate error accumulation within a single paradigm, we further propose an asymmetric co-training strategy with cross-guidance label generation. Finally, a confidence-based re-weighting strategy for label disambiguation is designed to enhance the model's tolerance to disambiguation failures. Extensive experiments on benchmark synthetic and real-world noisy datasets demonstrate the superiority of CA2C, which consistently outperforms previous state-of-the-art methods.

# Pipeline

![framework](figure1.jpg)

# Installation
```
pip install -r requirements.txt
```

# Datasets
We conduct noise robustness experiments on a synthetically corrupted dataset (i.e., CIFAR100N) and three real-world datasets (i.e., Web-Aircraft, Web-Car and Web-Bird).
Specifically, we create the noisy dataset CIFAR100N based on CIFAR100.
We adopt two classic noise structures: symmetric and asymmetric, with a noise ratio $n \in (0,1)$.

You can download the CIFAR10 and CIFAR100 on [this](https://www.cs.toronto.edu/~kriz/cifar.html).

You can download Web-Aircraft, Web-Car, and Web-Bird from [here](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset).

# Training

An example shell script to run CA2C on CIFAR100N :

```python
python main.py --gpu 7 --noise-type asymmetric --closeset-ratio 0.4 sgd --dataset cifar100nc --method CA2C 
```

# Results on CIFAR100N and CIFAR80N:

![framework](Table1.png)


# Results on Web-Aircraft, Web-Bird, and Web-Car:

![framework](Table2.png)

# Results on Food101N:

![framework](Table3.png)


# Effect of main modules in test accuracy (%) on CIFAR100N and CIFAR80N:

![framework](Table4.png)
