from distilabel.llms import TransformersLLM,OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts
from distilabel.steps import Step, StepInput, CombineColumns
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import TextGeneration, SelfInstruct, UltraFeedback
from typing import List
from pydantic import Field
import yaml


# Step 2: Load and parse the YAML file
with open('./config.yaml', 'r') as file:
    local_config = yaml.safe_load(file)

models = local_config['models']['arena-learning']
BATCH_SIZE = local_config['input-batch-size']

##################################
#         Arena Learning         #
##################################

class ArenaLearning:
    
    def __init__(self, initial_dataset, push_dataset, hf_token) -> None:
        
        self.initial_dataset = initial_dataset
        self.push_dataset = push_dataset
        self.HF_AUTH_TOKEN = hf_token

        with Pipeline(name="Battle of LLMs") as self.pipeline:
            self.load_dataset = LoadDataFromHub(
                name="dataset_for_arena_learning",
            )

            # first answer
            self.text_generation_1 = TextGeneration(
                name = "text_generation_01",
                llm = TransformersLLM(model=models['answer-gen'][0], device= "cuda:0"),
                # llm = TransformersLLM(model="Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct", device= "cuda:0"),
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
            )

            # second answer
            self.text_generation_2 = TextGeneration(
                name = "text_generation_02",
                # llm = TransformersLLM(model="mistralai/Mistral-7B-v0.1", device= "cuda:0"),
                llm = TransformersLLM(model=models['answer-gen'][1], device= "cuda:0"),
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
            )

            # third answer
            self.text_generation_3 = TextGeneration(
                name = "text_generation_04",
                llm = TransformersLLM(model=models['answer-gen'][2], device= "cuda:0"),
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
            )

            self.combine_columns = CombineColumns(
                name="self.combine_columns",
                columns=["generation", "model_name"],
                output_columns=["generations", "model_name"],
                input_batch_size=BATCH_SIZE
            )

            self.keep_columns_1 = KeepColumns(
                columns = ["instruction", "generations"]
            )

            self.clean = CleanGenerations(
                name="clean_generations"
            )

            self.ultrafeedback = UltraFeedback(
                llm=OpenAILLM(model=models['ranking'], api_key=local_config['gpt4-key']),
                input_batch_size=BATCH_SIZE,
                add_raw_output=False,
                aspect="overall-rating",
                output_mappings={"model_name": "ultrafeedback_model"}
            )

            self.best_gen = SelectBestGeneration(
                name="select_best_gen"
            )

            self.keep_columns = KeepColumns(
                columns=["instruction", "generation"]
            )

            self.load_dataset >> [self.text_generation_1, self.text_generation_2, self.text_generation_3] >> self.combine_columns >> self.keep_columns_1 >> self.clean >> self.ultrafeedback >> self.best_gen >> self.keep_columns

    def process(self):
        distiset_1 = self.pipeline.run(
                parameters={
                    self.load_dataset.name: {
                        "repo_id": self.initial_dataset,
                        "split": "train",
                    },
                    self.text_generation_1.name: {
                        "llm": {
                            "generation_kwargs": {
                                "temperature": self.config['temperature'],
                                "max_new_tokens": self.config['max-tokens'],
                            }
                        }
                    },
                    self.text_generation_2.name: {
                        "llm": {
                            "generation_kwargs": {
                                "temperature": self.config['temperature'],
                                "max_new_tokens": self.config['max-tokens'],
                            }
                        }
                    },
                    self.text_generation_3.name: {
                        "llm": {
                            "generation_kwargs": {
                                "temperature": self.config['temperature'],
                                "max_new_tokens": self.config['max-tokens'],
                            }
                        }
                    },
                    self.ultrafeedback.name: {
                        "llm": {
                            "generation_kwargs": {
                                "max_new_tokens": self.config['max-tokens'],
                                "temperature": self.config['temperature'],
                            }
                        }
                    },
                },
            )

        distiset_1.push_to_hub(
            self.push_dataset,
            token = self.HF_AUTH_TOKEN
        )


def clean_text(text):
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    return text

def clean_generations(generations):
    return [clean_text(item) for item in generations]


class CleanGenerations(Step):

    @property
    def inputs(self) -> List[str]:
        return ['generations']

    @property
    def outputs(self) -> List[str]:
        return ['generations']

    def process(self, inputs: StepInput) -> StepOutput:
        for input in inputs:
          input['generations'] = clean_generations(input['generations'])
        yield inputs

def select_best_generation(row):
    base_rating = row['ratings'][0]
    # print(base_rating)
    competitor_ratings = row['ratings'][1:]
    # print(competitor_ratings)
    base_answer = row['generations'][0]
    answers = row['generations'][1:]

    max_competitor_rating = max(competitor_ratings)
    # print(max_competitor_rating)
    if max_competitor_rating > base_rating:
        # print("in if")
        best_competitor_index = competitor_ratings.index(max_competitor_rating) + 1
        # print(best_competitor_index)
        row['generations'][0] = row['generations'][best_competitor_index]
        row['generations'][best_competitor_index] = base_answer
    row['generation'] = row['generations'][0]
    return row

class SelectBestGeneration(Step):

    @property
    def inputs(self) -> List[str]:
        return ['generations', 'ratings']

    @property
    def outputs(self) -> List[str]:
        return ['generation']

    def process(self, inputs: StepInput) -> StepOutput:
        for input in inputs:
          input = select_best_generation(input)
        yield inputs
