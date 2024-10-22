# Exam Questions Dataset and Evaluation

This project demonstrates the creation, annotation, and evaluation of an exam questions dataset using various tools and frameworks. The process involves generating exam questions, annotating them, creating a dataset, and evaluating language models on the task.

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

The `generate_dataset.py` script uses the Distilabel library to generate exam questions based on a Wikipedia article about transfer learning. It uses the Meta-Llama-3.1-8B-Instruct model to create questions, correct answers, and distractors.

To run the generation:

```sh
python generate_dataset.py
```

This will create a Distiset containing the generated exam questions.

### 2. Annotate Dataset

The `annotate_dataset.py` script takes the generated questions and creates an Argilla dataset for annotation. It sets up the dataset structure and populates it with the generated questions and answers.

To run the annotation process:

```sh
python annotate_dataset.py
```

This will create an Argilla dataset named "exam_questions" that can be used for manual review and annotation.

### 3. Create Dataset

The `create_dataset.py` script processes the annotated data from Argilla and creates a Hugging Face dataset. It handles both suggested and manually annotated answers.

To create the final dataset:

```sh
python create_dataset.py
```

This will push the dataset to the Hugging Face Hub under the repository "burtenshaw/exam_questions".

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

This command will evaluate the Zephyr-7B-beta model on the exam questions task and save the results in the "./evals" directory.

## Requirements

- Python 3.7+
- Distilabel
- Argilla
- Datasets
- LightEval
- Hugging Face Transformers

Make sure to install the required dependencies before running the scripts.

## Notes

- Ensure you have the necessary API keys and permissions set up for Hugging Face, Argilla, and any other services used in the scripts.
- The generation step uses the Meta-Llama-3.1-8B-Instruct model, which requires appropriate access and API keys.
- Review and adjust the Argilla server settings in the annotation script if needed.
- Make sure to handle any sensitive information (like API keys) securely and not expose them in your code or version control.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT License](https://opensource.org/licenses/MIT)
