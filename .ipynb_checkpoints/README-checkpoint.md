# Teaching Humans When To Defer to a Classifier via Examplars

This repository includes the code  and experiments  for our [paper Teaching Humans When To Defer to a Classifier via Examplars]() by Hussein Mozannar, Arvind Satyanarayan and David Sontag.


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

```
