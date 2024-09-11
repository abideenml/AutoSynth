from datasets import Dataset, load_dataset

dataset = load_dataset('csv', data_files=r'C:\Users\Muhammad Ahsan\Downloads\archive\test.csv')

dataset.push_to_hub(repo_id='title-content-dataset' , token='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')

