import huggingface_hub
card_data=huggingface_hub.ModelCardData(language='en',license='mit',library_name='pytorch')
card=huggingface_hub.ModelCard.from_template(card_data,
model_id="my-cool-model",
model_type="gan",
model_description="Classical GAN to generate celebA images",
developer="Hikmat Farhat",
repo="https://github.com/")
content=f"""
---
{card_data.to_yaml()}
---
"""
huggingface_hub.ModelCard(content)