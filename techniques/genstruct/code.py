from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts
from distilabel.steps.tasks import Genstruct
from distilabel.llms import TransformersLLM


class GenstructTechnique:

    def __init__(self, config) -> None:
        self.config = config
        self.hf_token = config['hf-token']
        
        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        self.instruct_model = next((model['path'] for model in config['models'] if model['type'] == 'instruct'), None)
                
        with Pipeline(name='genstruct-pipeline') as self.pipeline:
            self.load_data_from_hub = LoadDataFromHub(
                name='load-data-from-hub',
                output_mappings={'title': 'title', 'text': 'content'}
            )

            self.genstruct = Genstruct(
                llm=TransformersLLM(
                    model=self.instruct_model,
                    device='cuda:0'
                ),
                input_batch_size=self.config['input-batch-size'],
                output_mappings={"user": "instruction", "assistant": "response"}
            )

            self.keep_columns = KeepColumns(
                columns=['title', 'content', 'instruction', 'response']
            )

            self.load_data_from_hub >> self.genstruct >> self.keep_columns

    def process(self):
        distiset = self.pipeline.run(
            parameters={
                self.load_data_from_hub.name: {
                    'repo_id': self.input_dataset,
                    "split": "train"
                },
                self.genstruct.name: {
                    "llm": {
                        "generation_kwargs": {
                            "max_new_tokens": self.config['max-tokens'],
                            "temperature": self.config['temperature']
                        }
                    }
                }
            }
        )

        distiset.push_to_hub(
           self.output_dataset,
           token=self.hf_token
        )
