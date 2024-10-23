# Domain Specific Evaluation with Argilla, Distilabel, and LightEval

This project demonstrates the creation, annotation, and evaluation of a model on a domain specific task using Argilla, distilabel, and Lighteval. The process involves generating exam questions from multiple documents, annotating them, creating a dataset, and evaluating language models on the task.

This process is useful if you're applying language models to custom domains, and want to evaluate them on a task that is specific to that domain. Even if you're not fine-tuning models on this task, you can still use this process to evaluate how well they perform on your domain-specific task.

## Project Structure

```
domain-eval/
├── README.md
├── generate_dataset.py
├── annotate_dataset.py
├── create_dataset.py
├── evaluation_task.py
```

## Steps

### 1. Generate Dataset

The `generate_dataset.py` script uses the distilabel library to generate exam questions based on multiple text documents. It uses the specified model (default: Meta-Llama-3.1-8B-Instruct) to create questions, correct answers, and distractors. You should add you own data samples and you might wish to use a different model.

To run the generation:

```sh
python generate_dataset.py --input_dir path/to/your/documents --model_id your_model_id --output_path output_directory
```

This will create a Distiset containing the generated exam questions for all documents in the input directory.

### 2. Annotate Dataset

The `annotate_dataset.py` script takes the generated questions and creates an Argilla dataset for annotation. It sets up the dataset structure and populates it with the generated questions and answers, randomizing the order of answers to avoid bias. Once in Argilla, your or a domain expert can annotate the dataset with the correct answer.

To run the annotation process:

```sh
python annotate_dataset.py --dataset_path path/to/distiset --output_dataset_name argilla_dataset_name
```

This will create an Argilla dataset that can be used for manual review and annotation.

### 3. Create Dataset

The `create_dataset.py` script processes the annotated data from Argilla and creates a Hugging Face dataset. It handles both suggested and manually annotated answers. 

To create the final dataset:

```sh
python create_dataset.py --dataset_path argilla_dataset_name --dataset_repo_id your_hf_repo_id
```

This will push the dataset to the Hugging Face Hub under the specified repository.

### 4. Evaluation Task

The `evaluation_task.py` script defines a custom LightEval task for evaluating language models on the exam questions dataset. It includes a prompt function, a custom accuracy metric, and the task configuration.

## Running the Evaluation

To evaluate a model using LightEval with the custom exam questions task:

```sh
lighteval accelerate \
    --model_args "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    --tasks "community|exam_questions|0|0" \
    --custom_tasks domain-eval/evaluation_task.py \
    --output_dir "./evals"
```

This command will evaluate the specified model on the exam questions task and save the results in the "./evals" directory.

## Requirements

- Python 3.9+
- Distilabel
- Argilla
- Datasets
- LightEval
- Hugging Face Transformers

Make sure to install the required dependencies before running the scripts.

## Notes

- Ensure you have the necessary API keys and permissions set up for Hugging Face, Argilla, and any other services used in the scripts.
- The generation step uses the specified model (default: Meta-Llama-3.1-8B-Instruct), which may require appropriate access and API keys.
- Set the `HF_TOKEN` environment variable with your Hugging Face API token before running the generation script.
- Review and adjust the Argilla server settings in the annotation script if needed.
- Make sure to handle any sensitive information (like API keys) securely and not expose them in your code or version control.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](https://opensource.org/licenses/MIT)
