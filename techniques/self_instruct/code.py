from datasets import DatasetDict, Dataset
import pandas as pd
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import TextGeneration, SelfInstruct
from typing import List
from pydantic import Field
import yaml
 
class SelfInstructTechnique:

    def __init__(self, config) -> None:
        self.config = config

        self.hf_token = config['hf_token']

        self.MODELS = config['models']

        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        self.instruct_model = next((model['path'] for model in config['models'] if model['type'] == 'instruct'), None)
        self.response_model = next((model['path'] for model in config['models'] if model['type'] == 'response'), None)
        
        with Pipeline(name="Question Generation") as self.pipeline:
            self.load_hub_dataset = LoadDataFromHub(
                name="load_dataset",
                output_mappings={"prompt": "input"}
            )

            self.self_instruct = SelfInstruct(
                llm = TransformersLLM(
                    model=self.instruct_model, 
                    device= "cuda:0"
                ),
                input_batch_size=self.config['input-batch-size'],
                add_raw_output=False,
                num_instructions=self.config['n_instructions'],
                criteria_for_query_generation=criteria_for_query_generation,
                application_description=application_description,
                output_mappings={"model_name": "instruction_model"},
            )

            self.split_instr = SplitInstructions(
                name="split_instructions_step"
            )

            self.answer_generation = TextGeneration(
                llm = TransformersLLM(
                    model=self.response_model, 
                    device= "cuda:0"
                ),
                input_batch_size=self.config['input-batch-size'],
                add_raw_output=False,
                output_mappings={"generation": "response", "model_name": "response_model"},
            )

            self.keep_columns = KeepColumns(
                columns = ["input", "instruction", "response", "instruction_model", "response_model"]
            )

            self.load_hub_dataset >> self.self_instruct >> self.split_instr >> self.answer_generation >> self.keep_columns

    def process(self):

        distiset = self.pipeline.run(
            parameters={
                self.load_hub_dataset.name: {
                    "repo_id": self.input_dataset,
                    "split": "train",
                },
                self.self_instruct.name: {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": self.config['max-tokens'],
                            "temperature": self.config['temperature'],
                        },
                    },
                },
            },
        )
        distiset.push_to_hub(
           self.output_dataset,
           token=self.hf_token
        )

criteria_for_query_generation = (
    "1. Relevance: Ensure the questions are directly related to the content and context of the input paragraph."
    "2. Diversity: Include a variety of question types such as factual, analytical, inferential, and evaluative."
    "3. Clarity: Make sure each question is clear, concise, and unambiguous."
    "4. Complexity: Incorporate questions of varying difficulty levels, from simple recall to complex analysis."
    "5. Coverage: Cover the entire content of the paragraph, addressing different sections and key points."
    "6. Specificity: Frame questions to be specific and pointed, encouraging precise answers."
    "7. Engagement: Create questions that are interesting and engaging, promoting thoughtful responses."
    "8. Open-endedness: A portion of the generated questions should encourage creative and thoughtful responses, rather than simple factual recall."
    "9. Output: Provide only the five user queries without any introductory or explanatory text."
)

application_description = "This AI assistant is designed to generate a series of relevant and thought-provoking questions based on the provided context or input. The goal is to generate questions that cover different aspects of the topic without providing answers. The goal is to create an AI that can simulate human-like understanding and reasoning to respond to any query effectively."

# Defining Instruction Splitter class
class InstructionSplitter:
  def split_instructions_from_dataset(self, dataset: Dataset):
    new_rows = []
    for row in dataset:
      new_rows.extend(self.split_instructions_from_row(row))
    return new_rows

  def split_instructions_from_row(self, row):
      results = []
      for instruction in row['instructions']:
          result = row.copy()
          result['instruction'] = instruction
          del result['instructions']
          results.append(result)
      return results

class SplitInstructions(Step):
    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['instructions']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['instruction']

    def process(self, inputs: StepInput) -> StepOutput:
        inputs = InstructionSplitter().split_instructions_from_dataset(inputs)
        yield inputs