
from data_element_extractor import DataElementExtractor

dee = DataElementExtractor()

## Model ids used in the study
#model_id="openai/gpt-oss-20b"
#model_id="openai/gpt-oss-120b"
model_id="Qwen/Qwen3.6-27B"

dee.set_model(model_name=model_id, inference_type="transformers", model_type="Transformers", attn_implementation="eager", move_to_gpu=True, device_map="auto")
dee.load_topics("data/topics.json")
dee.show_topics_and_categories()
dee.set_choice_symbols("alphabetical")
dee.extract_from_table("data/trials-data-groundtruth.csv", constrained_output=False, with_evaluation=True, batch_size=1)




