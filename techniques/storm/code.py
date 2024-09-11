from pydantic import Field
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub
import yaml
from techniques.utilities import InstructClassification, FilterOnRanking

with open('./config.yaml') as file:
    config = yaml.safe_load(file)

DATASETS = config['datasets']
MODELS = config['models']
LABELS = config['labels']
THRESHOLDS = config['thresholds']

class StormTechnique:

    def __init__(self, hf_token) -> None:
        
        self.HF_AUTH_TOKEN = hf_token

        with Pipeline(name="Data Curation") as self.curation_pipeline:
            self.load_dataset = LoadDataFromHub(
                name="Load_Dataset",
                output_mappings={"instruction": "question", "generation": "answer"}
                )

            self.educative_classifier = InstructClassification(
                name="Educative_Classifier",
                candidate_labels=LABELS['educative'],
                input_batch_size=config['batch-size'],
                model=MODELS['classification']
            )

            self.educative_filter = FilterOnRanking(
                name="Educative_Filter",
                threshold=THRESHOLDS['educative'],
                output_mappings={"ranking": "educative"}
            )

            self.difficulty_classifier = InstructClassification(
                name="Difficulty_Classifier",
                candidate_labels=LABELS['difficulty'],
                input_batch_size=config['batch-size'],
                model=MODELS['classification']
            )

            self.difficulty_filter = FilterOnRanking(
                name="Difficulty_Filter",
                threshold=THRESHOLDS['difficulty'],
                output_mappings={"ranking": "difficulty"}
            )

            self.load_dataset >> self.educative_classifier >> self.educative_filter >> self.difficulty_classifier >> self.difficulty_filter

    def process(self):
        curated_data = self.curation_pipeline.run(
            parameters={
                self.load_dataset.name: {
                    "repo_id": DATASETS['initial'],
                    "split": "train",
                },
            },
        )

        curated_data.push_to_hub(
            DATASETS['curated-push'],
            token = self.HF_AUTH_TOKEN
        )
