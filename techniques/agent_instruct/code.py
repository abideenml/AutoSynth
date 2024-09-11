from datasets import DatasetDict, Dataset
import pandas as pd
from distilabel.llms import TransformersLLM,OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts
from distilabel.steps import Step, StepInput, CombineColumns
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import TextGeneration, SelfInstruct, UltraFeedback
from typing import List
from pydantic import Field
from techniques.utilities import *
import yaml

with open('./config.yaml', 'r') as file:
    local_config = yaml.safe_load(file)

models = local_config['models']
question_prompt = local_config.get('question_gen_prompt')
suggest_prompt = local_config.get('suggestion_gen_prompt')
refined_question_prompt = local_config.get('refined_question_gen_prompt')
BATCH_SIZE = local_config['input-batch-size']

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


##################################
#         Agent Instruct         #
##################################

class AgentInstruct:

    def __init__(self, initial_dataset, push_dataset, hf_token):
        self.initial_dataset = initial_dataset
        self.push_dataset = push_dataset
        self.HF_AUTH_TOKEN = hf_token

        shared_model = TransformersLLM(model=models['agent-instruct'], device="cuda:0")


        with Pipeline(name="Question Generation") as self.pipeline:
            self.load_hub_dataset = LoadDataFromHub(
                name="load_dataset",
                output_mappings={"prompt": "instruction"}
            )

            self.text_generation = TextGeneration(
                # llm = TransformersLLM(model="microsoft/Phi-3-mini-4k-instruct"),
                # llm = TransformersLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", device= "cuda:0"),
                llm = shared_model,
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
                output_mappings={"generation": "input", "model_name": "transformed_text_model"},
            )

            self.self_instruct = SelfInstruct(
                llm = shared_model,
                # llm = TransformersLLM(model="Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct", device= "cuda:0"),
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
                num_instructions=5,
                criteria_for_query_generation=criteria_for_query_generation,
                application_description=question_prompt,
                output_mappings={"model_name": "instructions_model"},
            )

            self.rename_1 = RenameColumn(
                name="rename_instr_to_raw_seed",
                old_column="instruction",
                new_column="raw_seed"
            )

            self.split_instr = SplitInstructions(
                name="split_instructions_step"
            )

            self.prompt_change_1 = ReplaceAllColumnValues(
                name="suggestion_system_prompt",
                column_name="system_prompt",
                new_value=suggest_prompt
            )

            self.suggestion_generation = TextGeneration(
                # llm = TransformersLLM(model="meta-llama/Meta-Llama-3-8B-Instruct", device= "cuda:0"),
                llm = shared_model,
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
                output_mappings={"generation": "suggestions", "model_name": "suggestions_model"},
            )

            self.rename_2 = RenameColumn(
                name="rename_instr_to_question",
                old_column="instruction",
                new_column="question"
            )

            self.merge_question_suggestions = MergeQuestionSuggesions(
                name="merge_question_suggestions_step"
            )

            self.prompt_change_2 = ReplaceAllColumnValues(
                name="question_system_prompt",
                column_name="system_prompt",
                new_value=refined_question_prompt
            )

            self.question_generation = TextGeneration(
                llm = shared_model,
                # llm = TransformersLLM(model="Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct", device= "cuda:0"),
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
                output_mappings={"model_name": "refined_q_model"},
            )

            self.keep_columns = KeepColumns(
                columns=["generation"],
            )

            self.load_hub_dataset >> self.text_generation >> self.self_instruct >> self.rename_1 >> self.split_instr >> self.prompt_change_1 >> self.suggestion_generation >> self.rename_2 >> self.merge_question_suggestions >> self.prompt_change_2 >> self.question_generation >> self.keep_columns


    def process(self):
        distiset = self.pipeline.run(
            parameters={
                self.load_hub_dataset.name: {
                    "repo_id": self.initial_dataset,
                    "split": "train",
                },
                self.text_generation.name: {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": local_config['max-tokens'],
                            "temperature": local_config['temperature'],
                        },
                    },
                },
                self.self_instruct.name: {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": local_config['max-tokens'],
                            "temperature": local_config['temperature'],
                        },
                    },
                },
                self.suggestion_generation.name: {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": local_config['max-tokens'],
                            "temperature": local_config['temperature'],
                        },
                    },
                },
                self.question_generation.name: {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": local_config['max-tokens'],
                            "temperature": local_config['temperature'],
                        },
                    },
                },
            },
        )

        # Map the function to split 'generation' column
        split_dataset = distiset['default']['train'].map(
            split_generation,
            batched=True,
            remove_columns=['generation'],
            batch_size=1
        )


        # Pushing dataset after question generation to huggingface
        split_dataset.push_to_hub(
            self.push_dataset,
            token = self.HF_AUTH_TOKEN
        )


#################################
# Defining Useful Functions     #
#################################

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

class MergeQuestionSuggesions(Step):
    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['question', 'suggestions']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['instruction']

    def process(self, inputs: StepInput) -> StepOutput:
        for example in inputs:
          combined_text = example['question'] + "\n\nSuggestions:\n" + example['suggestions']
          example['instruction'] = combined_text
        yield inputs

# Define the function to split the 'generation' column into multiple rows
def split_generation(examples):
    # Ensure examples is a dictionary with lists
    generations = examples['generation']

    # Process each entry in the batch
    new_examples = []
    for generation in generations:
        questions = generation.split('?')
        questions = [q.strip() + '?' for q in questions if q.strip()]
        for question in questions:
            new_example = {
                'instruction': question,
                **{k: v for k, v in examples.items() if k != 'generation'}
            }
            new_examples.append(new_example)

    return {'instruction': [e['instruction'] for e in new_examples]}
