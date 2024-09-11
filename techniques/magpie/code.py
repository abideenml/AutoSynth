from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import Magpie
from typing import List
from pydantic import Field
import yaml
from techniques.utilities import SplitList

with open('./config.yaml') as file:
    config = yaml.safe_load(file)

DATASETS = config['datasets']
MODELS = config['models']

class MagpieTechnique:
   
    def __init__(self, hf_token) -> None:
        self.hf_token = hf_token

        with Pipeline(name="Question Generation") as self.pipeline:
            self.load_hub_dataset = LoadDataFromHub(
                name="load_dataset",
            )

            self.magpie = Magpie(
                llm=TransformersLLM(
                    model=config['model'],
                    magpie_pre_query_template=config['magpie_pre_query_template'],
                    generation_kwargs={
                        "temperature": config['temperature'],
                        "max_new_tokens": config['max_tokens'],
                    },
                    device="cuda:0",
                ),
                # only_instruction=True,
                n_turns = config['conversation_turns']
            )

            self.instruct_res_pairs = ConversationToInstructPairs(
                name="conversation_to_instruct_pairs",
            )

            self.split_instructions = SplitList(
                name="split_instructions",
                split_column = 'instructions',
                output_mappings={"splitted": "instruction"}
            )

            self.split_responses = SplitList(
                name="split_responses",
                split_column = 'responses',
                output_mappings={"splitted": "response"}
            )

            self.keep_columns = KeepColumns(
                columns = ["instruction", "response"]
            )

            self.load_hub_dataset >> self.magpie >> self.instruct_res_pairs >> self.split_instructions >> self.split_responses >> self.keep_columns

    def process(self):
        distiset = self.pipeline.run(
            parameters={
                self.load_hub_dataset.name: {
                    "repo_id": DATASETS['initial'],
                    "split": "train",
                },
            },
        )

        distiset.push_to_hub(
           DATASETS['push'],
           token=self.hf_token
        )


class ConversationToInstructPairs(Step):
    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['conversation']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['instructions', 'responses']

    def process(self, inputs: StepInput) -> StepOutput:
        for example in inputs:
          instructions = []
          responses = []
          for talk in example['conversation']:
            if talk['role'] == 'user':
              instructions.append(talk['content'])
            else:
              responses.append(talk['content'])
          example['instructions'] = instructions
          example['responses'] = responses
          example.pop('conversation')
        yield inputs