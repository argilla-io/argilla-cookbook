# argilla-cookbook

This repository contains simple examples using Argilla tools to build AI.

## Retrieval Augmented Generation (RAG) with Argilla

These examples demonstrate the use of Argilla tools for retrieval-augmented generation (RAG) tasks. The notebooks showcase the simple ways of using Argilla to improve retrieval accuracy and model performance in question-answering tasks.



| Notebook                               | Description                                                                                                                                                                                                                              |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [dspy_agent_arxiv_tools_prompt_optimization.ipynb](./dspy_agent_arxiv_tools_prompt_optimization.ipynb) | This notebook, titled **"ArXiv Tools for Prompt Optimization"**, is designed to assist in optimizing and refining prompts for use in agentic applications. It includes tools for prompt evaluation, benchmarking, and iterative improvement, using arXiv datasets and models. |
| [fewshot_classify_langchain.ipynb](./fewshot_classify_langchain.ipynb)    | This notebook demonstrates few-shot classification techniques using **LangChain**, focusing on building and evaluating language model-driven classification tasks with minimal labeled examples.                                              |
| [multihop_langchain_frames_benchmark.ipynb](./multihop_langchain_frames_benchmark.ipynb) | This notebook benchmarks multi-hop reasoning using **LangChain** with frame-based models, leveraging the **Google Frames Benchmark** dataset. It evaluates multi-step query resolution tasks, using **Argilla** to review and improve model outputs.                            |
| [rag_monitor_llamaindex.ipynb](./rag_monitor_llamaindex.ipynb)        | This notebook demonstrates the use of retrieval-augmented generation (RAG) with **LlamaIndex** for monitoring and optimizing the retrieval process. It focuses on improving retrieval accuracy and model performance in question-answering tasks. |
| [rag_benchmark_compare_llms_ragas_haystack.ipynb](./rag_benchmark_compare_llms_ragas_haystack.ipynb)        | This notebook demonstrates how to benchmark and compare large language models (LLMs) in a Retrieval-Augmented Generation (RAG) pipeline using **Haystack** for RAG, **Ragas** for evaluation, and **Argilla** for monitoring. It guides users through setting up a RAG pipeline with the **PubMedQA_instruction** dataset, comparing two LLMs (Microsoft's Phi-3.5-mini-instruct and Meta's Llama-3.1-8B-Instruct), and evaluating performance based on metrics like faithfulness, relevancy, and correctness. Results are logged and analyzed in Argilla to identify the best-performing LLM. Alternatives for RAG and evaluation tools are also suggested.. |

## Labeling Datasets in Argilla

These examples demonstrate the use of tools for labeling in Argilla dataset. The notebooks showcase the simple ways of using Argilla to label datasets with LLMs.

| Notebook                               | Description                                                                                                                                                                                                                              |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [label_argilla_datasets_with_llms_annotation_guidelines_and_distilabel.ipynb](./label_argilla_datasets_with_llms_annotation_guidelines_and_distilabel.ipynb) | This notebook demonstrates how to label datasets with LLMs using Argilla based on the written fields, questions and annotation guidelines. It uses the **ArgillaLabeller** class from **distilabel** library. This class will use  will use an LLM to label the datasets. These labels will then be converted into **Suggestion** objects and added to the records. |
| [efficient_zero_shot_token_classification_with_gliner_spanmarker.ipynb](./efficient_zero_shot_token_classification_with_gliner_spanmarker.ipynb) | This notebook demonstrates how to use **GLiNER** and **SpanMarker** for efficiently starting projects with smaller models. |
| [efficient_zero_shot_text_classification_with_setfit.ipynb](./efficient_zero_shot_text_classification_with_setfit.ipynb) | This notebook demonstrates how to use **SetFit** for efficiently starting projects with smaller models. |


