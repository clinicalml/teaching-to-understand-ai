# Teaching Humans When To Defer to a Classifier via Examplars

This repository includes the code  and experiments  for our [paper Teaching Humans When To Defer to a Classifier via Examplars](https://arxiv.org/abs/2111.11297) by Hussein Mozannar, Arvind Satyanarayan and David Sontag.


**This repository is currently being expanded.**


The use of our algorithm is pretty simple:

```python
from teaching import TeacherExplainer
# prepare data
teacher = TeacherExplainer(data_x, data_y, human_predictions, AI_predictions, prior_rejector, kernel, metric_y, alpha, number_teaching_points)
teaching_x, teaching_gammas, teaching_labels, teaching_indices = teacher.get_teaching_examples()
```

Refer to the source [teaching.py](teaching.py) for the correct specification of the arguments and outputs which should be:

```
Args:
    data_x: 2d numpy array of the features
    data_y: 1d numpy array of labels
    human_predictions:  1d array of the human predictions 
    AI_predictions:  1d array of the AI predictions 
    prior_rejector: 1d binary array of the human prior rejector preds 
    kernel: function that takes as input two inputs and returns a positive number, plays the role of K(.,.)
    metric_y: metric function (positive, the higher the better) between predictions and ground truths, must behave like rbf_kernel from sklearn
    alpha: parameter of selection algorithm, 0 for double greedy and 1 for consistent radius
    number_teaching_points: number of teaching points to return
Returns:
    teaching_x: 2d numpy array of teaching points features
    teaching_indices: indices of the teaching points in self.data_x
    teaching_gammas: 1d numpy of gamma values used
    teaching_labels: 1d array of deferral labels where 1 signifies defer to AI and 0 signifies don't defer to AI
```

For an example usage please the following [notebook](test_teaching.ipynb)

Quick Links:
* [CIFAR](cifar/README.md)
* [Synthetic Data](synthetic/README.md)
* [HotpotQA](hotpotqa/README.md)
* [User Study Details](userstudy/README.md)
* [SAE error analysis](SAE_error/README.md)

This repository is currently being expanded.

## Abstract
Expert decision makers are starting to rely on data-driven automated agents to assist them with various tasks. For this collaboration to perform properly, the human decision maker must have a mental model of when and when not to rely on the agent. In this work, we aim to ensure that human decision makers learn a valid mental model of the agent's strengths and weaknesses. To accomplish this goal, we propose an exemplar-based teaching strategy where humans solve a set of selected examples and with our help generalize from them to the domain. We present a novel parameterization of the human's mental model of the AI that applies a nearest neighbor rule in local regions surrounding the teaching examples. Using this model, we derive a near-optimal strategy for selecting a representative teaching set. We validate the benefits of our teaching strategy on a multi-hop question answering task with an interpretable AI model using crowd workers. We find that when workers draw the right lessons from the teaching stage, their task performance improves. We furthermore validate our method on a set of synthetic experiments. 

This repository contains multiple jupyter notebooks


## Requirements

*Ongoing* We will  include a [requirements file](requirements.txt) that covers everything required to run the notebooks from a new environment.


## Citation



```
@misc{mozannar2021teaching,
      title={Teaching Humans When To Defer to a Classifier via Examplars}, 
      author={Hussein Mozannar and Arvind Satyanarayan and David Sontag},
      year={2021},
      eprint={2111.11297},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
