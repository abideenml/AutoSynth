from datasets import Dataset

# Your data
data = [
    {
        "system_prompt": "You are a helpful AI assistant. The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions. Your goal is to provide thorough, relevant, and insightful responses to help the user with their queries."
    },
    {
        "system_prompt": "You are a helpful AI assistant. The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions. Your goal is to provide thorough, relevant, and insightful responses to help the user with their queries."
    },
]

# Convert the data to a Hugging Face Dataset
dataset = Dataset.from_list(data)

# Push to Hugging Face Hub
dataset.push_to_hub("ahsanirfan961/magpie-initial", token="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")
