from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch  # no longer needed if not loading weights
from src.News_Summarizer.summarization import T5smallFinetuner
# from src.News_Summarizer.aws_s3 import download_model_from_s3

# Load tokenizer and base model (pretrained weights from HuggingFace hub)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
base_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Use the untrained model as a placeholder
prediction_model = T5smallFinetuner(model=base_model, tokenizer=tokenizer)

# Skip downloading from S3 and skip loading local .pt file
# download_model_from_s3('model-news','t5-small.pt',"artifacts/t5-small.pt")
# state_dict = torch.load('artifacts/t5-small.pt', weights_only=True)
# prediction_model.load_state_dict(state_dict['model_state_dict'])
# prediction_model.eval()
