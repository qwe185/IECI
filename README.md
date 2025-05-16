# IECI
### Brief Introduction
In this study, we present IECI, a pipelined framework for ECI.The framework consists of two subtasks:PRED (Prompt Enhancement and Semantic-based Event Detection): for the event detection task. The model combines contextual encoding of semantic roles with cue templates to direct the model's attention to key semantic segments in a sentence, thus improving the accuracy and robustness of event type recognition. SRCIG (Semantic Role-Guided Graph-Based Causal Reasoning): is used for document-level event causality recognition. The model combines semantic role information to construct causal graphs and introduces a dynamic threshold adjustment mechanism to achieve more semantically oriented causal inference. The model optimizes the causal representation of events iteratively through the graph structure, gradually revealing potential causal chains and enhancing its ability to capture complex causal relationships.
## PRED
### dataset
Dataset storage path: PRED/data/ESL/data_ esl
### model
This project uses the Roberta large model provided by HuggingFace. https://huggingface.co/FacebookAI/roberta-large
### test configuration
If only testing is conducted, please modify the configuration or parameters to set the following variables to True:<br>
inference_only = True<br>
single = True
## SRCIG
### model
This project uses the bert-base-uncased model provided by HuggingFace. https://huggingface.co/google-bert/bert-base-uncased/tree/main
### dataset
EventStoryLine v0.9： https://github.com/tommasoc80/EventStoryLine/
## Usage
You can download this program to run it on your device.
