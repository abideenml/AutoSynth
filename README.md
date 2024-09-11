# AutoSynth

Automatically create synthetic data using SOTA techniques (Self Instruct, Magpie, Agent Instruct, Arena Learning, Genstruct, Instruction Synthesizer, Self-Curation) using your LLMs.

```bash

## Research log

11-09-2024
----------
implemented [Instruction Synthesizer](./techniques/instruction_synthesizer/) 
using the instruction-synthesizer pretrained model from huggingface hub. Defined custom pipeline 
steps to generate multiple instruction response pairs from a single text corpus. Fixed the 
multiprocessing conflicts arising when using external models using transformers library in the custom Step sub-classes.

10-09-2024
----------
implemented [Genstruct](./techniques/genstruct/) technique using the provided distilabel task. 
Used the Genstruct-7B pretrained models to generate instruction-response pairs from raw text corpora.

31-08-2024
----------
implemented [Magpie](./techniques/magpie/) technique for conversation generation using the 
provided distilabel task. Defined custom distilabel pipeline step to perform list 
splitting and converting conversation to instruction-response pairs.

30-08-2024
----------
implemented Storm technique for Data Curation as described in this [paper](https://huggingface.co/blog/akjindal53244/llama31-storm8b). 
Used strong models like Meta Llama and Facebook Bert Classifier to separate quality data.

26-08-2024
----------
implemented [Self Instruct](./techniques/self_instruct/) technique. Coded custom distilabel 
pipeline steps to perform tasks like renaming columns or swapping columns during pipeline processing. 
Used the distilabel Self-Instruct task to generate n instructions from a raw text corpora.

24-08-2024
----------
implemented [Arena Learning](./techniques/arena_learning/). Used Phi-3 mini, TinyLlama, OpenAI GPT 4o 
mini for testing. It generates multiple answers to a question using different models 
and then ranks them using a stronger model. Used GPT 4o mini for ranking.

22-08-2024
----------
implemented [Agent Instruct](./techniques/agent_instruct/). Used Phi-3 mini for testing. It generates instructions using multiple agents.
```

## ✅ Techniques
1. Self Instruct
2. Magpie
3. Agent Instruct with Arena Learning
4. Storm 
5. Genstruct
6. Instruction Synthesizer

## Setup
**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```
**2. Setup configs:**

The directory of each technique in [techniques](./techniques/) contains an example config.yaml file. Adjust the values of datasets, models, and other parameters in that file.

**3. Run the main.py script:**
Specify the config.yaml file path and the technique name while runnning this command.
```bash
python main.py --config=./techniques/genstruct/config.yaml --technique=genstruct
```

For help run,
```bash
python main.py --help
```

## 🦋 Citation

If you find this code useful, please cite the following:

```
@misc{Zain2024AutoSynth,
  author = {Zain, Abideen},
  title = {AutoSynth},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/abideenml/AutoSynth}},
}
```

## Connect with me

If you'd love to have some more AI-related content in your life :nerd_face:, consider:

* Connect and reach me on [LinkedIn](https://www.linkedin.com/in/zaiinulabideen/) and [Twitter](https://twitter.com/zaynismm)
* Follow me on 📚 [Medium](https://medium.com/@zaiinn440)
* Check out my 🤗 [HuggingFace](https://huggingface.co/abideen)