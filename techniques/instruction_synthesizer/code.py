from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, KeepColumns, LoadDataFromDicts
from distilabel.steps.tasks import Genstruct
from distilabel.llms import TransformersLLM
from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from typing import List
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class InstructionSynthesizerTechnique:

    def __init__(self, config):
        self.config = config
        self.hf_token = config['hf-token']
        self.input_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'input'), None)
        self.output_dataset = next((dataset['path'] for dataset in config['datasets'] if dataset['type'] == 'output'), None)

        self.instruct_model = next((model['path'] for model in config['models'] if model['type'] == 'instruct'), None)

        with Pipeline(name='synthesizer-pipeline') as self.pipeline:
            self.load_data_from_hub = LoadDataFromHub(
                name='load-data-from-hub',
            )

            self.synthesizer = InstructionSynthesizer(
                name='synthesizer',
                model_path=self.instruct_model
            )

            self.keep_columns = KeepColumns(
                columns=['instruction', 'response']
            )

            self.load_data_from_hub >> self.synthesizer >> self.keep_columns


    def process(self):
        distiset = self.pipeline.run(
            parameters={
                self.load_data_from_hub.name: {
                    'repo_id': self.input_dataset,
                    "split": "train"
                },
            },
        )

        distiset.push_to_hub(
            self.output_dataset, 
            token=self.hf_token
        )

def parse_pred(pred):
    """Extract the list of instruction-response pairs from the prediction"""
    QA_str_list = pred.split('</END>')
    if not pred.endswith('</END>'):
        QA_str_list = QA_str_list[:-1]

    QA_list = []
    raw_questions = []
    for QA_str in QA_str_list:
        try:
            assert len(QA_str.split('<ANS>')) == 2, f'invalid QA string: {QA_str}'
            Q_str, A_str = QA_str.split('<ANS>')
            Q_str, A_str = Q_str.strip(), A_str.strip()
            assert Q_str.startswith('<QUE>'), f'invalid question string: {Q_str} in QA_str: {QA_str}'
            assert len(A_str) > 0, f'invalid answer string in QA_str: {QA_str}'
            Q_str = Q_str.replace('<QUE>', '').strip()
            assert Q_str.lower() not in raw_questions, f'duplicate question: {Q_str}'
            QA_list.append({'Q': Q_str, 'A': A_str})
            raw_questions.append(Q_str.lower())
        except:
            pass

    return QA_list

def get_instruction_response_pairs(context, model, tokenizer):
    '''Prompt the synthesizer to generate instruction-response pairs based on the given context'''
    prompt = f'<s> <CON> {context} </CON>\n\n'
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids=inputs, max_new_tokens=400, do_sample=False)[0]

    pred_start = int(inputs.shape[-1])
    pred = tokenizer.decode(outputs[pred_start:], skip_special_tokens=True)
    return parse_pred(pred)
    
class InstructionSynthesizer(Step):

    model_path: str = Field(..., description="Path to the model")

    def __init__(self, name: str):
        super().__init__(name=name)
        self._model = None
        self._tokenizer = None
    
    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['text']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['instruction', 'response']

    def process(self, inputs: StepInput) -> StepOutput:
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_path).to('cuda')
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        results = []
        for example in inputs:
            context = example['text']
            instruction_response_pairs = get_instruction_response_pairs(context, self._model, self._tokenizer)
            print(instruction_response_pairs)
            text_index = inputs.index(example)
            for pair in instruction_response_pairs:
                results.append({"text": context, "instruction": pair["Q"], "response": pair["A"]})
        yield results