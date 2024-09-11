from distilabel.llms import TransformersLLM,OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts
from distilabel.steps import Step, StepInput, CombineColumns
from distilabel.steps.typing import StepOutput
from distilabel.steps.tasks import TextGeneration, SelfInstruct, UltraFeedback
from typing import List
from pydantic import Field
import yaml
from techniques.agent_instruct.code import AgentInstruct
from techniques.arena_learning.code import ArenaLearning


# Step 2: Load and parse the YAML file
with open('./config.yaml', 'r') as file:
    local_config = yaml.safe_load(file)

DATASETS = local_config['datasets']

class AgentArenaTechnique:
    
    def __init__(self, hf_token) -> None:
        self.HF_AUTH_TOKEN = hf_token

        self.agent_instruct = AgentInstruct(DATASETS['initial'], DATASETS['instruction-response-push'], hf_token)
        self.arena_learning = ArenaLearning(DATASETS['instruction-response-push'], DATASETS['instructions-push'], hf_token)
    
    def process(self):
        self.agent_instruct.process()
        self.arena_learning.process()