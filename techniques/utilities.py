from distilabel.steps import Step, StepInput
from distilabel.steps.typing import StepOutput
from typing import List
from pydantic import Field
from transformers import pipeline
from datasets import load_dataset, Dataset

class RenameColumn(Step):

    old_column: str = Field(..., description="The name of the column to rename.")
    new_column: str = Field(..., description="The new name for the column.")

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return [self.old_column]

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return [self.new_column]

    def process(self, inputs: StepInput) -> StepOutput:
        for example in inputs:
            if self.old_column in example:
                example[self.new_column] = example.pop(self.old_column)  # Rename the column
        yield inputs

class ReplaceAllColumnValues(Step):
    column_name: str = Field(..., description="The name of the column whose values will be changed.")
    new_value: str = Field(..., description="The new value that will replace all existing values in the column.")

    @property
    def inputs(self) -> List[str]:
        return [self.column_name]

    @property
    def outputs(self) -> List[str]:
        return [self.column_name]

    def process(self, inputs: StepInput) -> StepOutput:
        for example in inputs:
            if self.column_name in example:
                example[self.column_name] = self.new_value  # Update the column value
        yield inputs

class InstructClassification(Step):

    candidate_labels: List = Field(..., description="The list containing the labels to be classififed")
    model: str = Field(..., description="The model to be used for classification")

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['question', 'answer']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['ranking']

    def process(self, inputs: StepInput) -> StepOutput:
        classifier = pipeline('zero-shot-classification', model=self.model, device=0)
        for example in inputs:
            sequence = f"Q. {example['question']}\nA. {example['answer']}"
            prediction = classifier(sequence, self.candidate_labels)['scores']
            example['ranking'] = str(prediction.index(min(prediction)) + 1)
        yield inputs

class FilterOnRanking(Step):

    threshold: int = Field(..., description="All the instructions with ranking >= threshold shall be considered")

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return ['question', 'answer', 'ranking']

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['question', 'answer', 'ranking']

    def process(self, inputs: StepInput) -> StepOutput:
        inputs = [example for example in inputs if int(example['ranking']) >= self.threshold]
        yield inputs


# Defining Instruction Splitter class
class ListSplitter:

  split_column = 'instructions'

  def __init__(self, split_column) -> None:
     self.split_column = split_column

  def split_instructions_from_dataset(self, dataset: Dataset):
    new_rows = []
    for row in dataset:
      new_rows.extend(self.split_instructions_from_row(row))
    return new_rows

  def split_instructions_from_row(self, row):
      results = []
      for instruction in row[self.split_column]:
          result = row.copy()
          result['splitted'] = instruction
          del result[self.split_column]
          results.append(result)
      return results

class SplitList(Step):

    split_column: str = Field(..., description="The column containing list to split")

    @property
    def inputs(self) -> List[str]:
        # Specify the input fields expected by this step
        return [self.split_column]

    @property
    def outputs(self) -> List[str]:
        # Specify the output fields that this step will produce
        return ['splitted']

    def process(self, inputs: StepInput) -> StepOutput:
        inputs = ListSplitter(self.split_column).split_instructions_from_dataset(inputs)
        yield inputs