DEFAULT_NLP_SEQUENCE_CLASSIFICATION_MODEL = "distilbert-base-uncased"
DEFAULT_NLP_SUMMARIZATION_MODEL = "distilbert-base-uncased"
DEFAULT_TABULAR_MODEL = "distilbert-base-uncased"
DEFAULT_CV_IMAGE_CLASSIFICATION_MODEL = "google/vit-base-patch32-224-in21k"
DEFAULT_CV_OBJECT_DETECTION_MODEL = "facebook/detr-resnet-101"
NLP_PRETRAINED_MODELS = [
    "bert-base-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "distilbert-base-cased",
    "distilbert-base-uncased",
    "xlm-roberta-base",
    "xlm-roberta-large",
]

CV_PRETRAINED_MODELS = [
    "google/vit-base-patch16-224-in21k",
    "google/vit-base-patch16-384",
    "google/vit-base-patch32-224-in21k",
    "google/vit-base-patch32-384",
    "google/vit-large-patch16-224-in21k",
    "google/vit-large-patch16-384",
    "google/vit-large-patch32-224-in21k",
    "google/vit-large-patch32-384",
]
IMPORT_ERROR_MESSAGE = (
    "To enable embedding generation, the arize module must be installed with "
    "extra dependencies. Run: pip install 'arize[AutoEmbeddings]'."
)

GPT = "GPT"
BERT = "BERT"
VIT = "ViT"
