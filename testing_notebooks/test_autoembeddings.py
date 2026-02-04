import pandas as pd

from arize.embeddings import EmbeddingGenerator, UseCases

print(EmbeddingGenerator.list_pretrained_models())

df = pd.DataFrame(
    {
        "text": [
            "Hello world.",
            "Artificial Intelligence is the future.",
            "OpenAI creates powerful AI models.",
        ],
    }
)
generator = EmbeddingGenerator.from_use_case(
    use_case=UseCases.NLP.SEQUENCE_CLASSIFICATION,
    model_name="distilbert-base-uncased",
    tokenizer_max_length=512,
    batch_size=100,
)
df["text_vector"] = generator.generate_embeddings(text_col=df["text"])
