# SAE model error analysis experiments

We include a notebook to reproduce our [SAE](https://github.com/JD-AI-Research-Silicon-Valley/SAE) model error analysis found in Appendix C.

There is two steps:

- get the predictions of the SAE model on the val set using [sae_predict_hotpot.ipynb](sae_predict_hotpot.ipynb), we've already done this step for you and the predictions are in [predictions_ans_val.json](predictions_ans_val.json)

- the notebook [hotpot_qa_ai_analysis.ipynb](hotpot_qa_ai_analysis.ipynb) replicates the analysis with additional plots.