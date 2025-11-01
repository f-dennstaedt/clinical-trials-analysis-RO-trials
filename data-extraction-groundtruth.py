
from data_element_extractor import DataElementExtractor

dee = DataElementExtractor()
dee.set_model(model_name="rombodawg/Rombos-LLM-V2.6-Qwen-14b", inference_type="transformers", model_type="Transformers", attn_implementation="eager", move_to_gpu=True, device_map="cuda:1")
dee.load_topics("data/topics.json")
dee.show_topics_and_categories()
dee.set_choice_symbols("alphabetical")
dee.extract_from_table("data/trials-data-groundtruth.csv", with_evaluation=True, batch_size=1)