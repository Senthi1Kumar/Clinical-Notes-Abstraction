# Clinical Notes Abstraction

This project develops an Named Entity Recognition (NER) pipeline for medical text processing, leveraging state-of-the-art ML techniques to extract and classify medical entities from unstructured clinical notes. On top of that to build a fine-tuned model for abstractive summarization.

## About Dataset

The dataset is acquired from from HF here: [AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) which is curated by *Antoine Bonnet* and *Paul Boulenger*.

This dataset is an extension of existing datasets from various sources such as Real clinical notes, Synthetic dialogues, and Structured patient information. It contains 30K rows, and fields namely: `idx`, `note`, `full_note`, `conversation`, `summary`.

## NER pipeline

The Named Entity Recognition model, here I used is [gliner_small-v2.5](https://huggingface.co/gliner-community/gliner_small-v2.5) from GLINER community, it classifies and categorizes any type of entities using BERT-based transformer.

This small model extracted entities from `note` attribute in the table into different type of medications, diagnosis, symptoms, procedures, and body parts with evaluation.

## Data Transformation Exploration

- Initial setup: dbt-core with PostgreSQL adatper
- Investigated mulitple data transformation approaches
- Experimentation with:
    - Local PostgreSQL itegration
    - Python models (for pre-processing)
    - Explored BigQuery platform limitations

## Fine-tuning roadmap [WIP]

- Planned implementation of SLM fine-tuning
- Focusing on abstractive summarization techniques

### Future Enhancements

- Implement robust data transformation pipeline.
- Enhance NER model precision and recall.
