import torch
from openpi.models import pi0_model # Import the specific model class

# 1. Load your fine-tuned checkpoint
checkpoint_path = "experiments/your_custom_run/checkpoints/step_80000.ckpt"
model = pi0_model.load_from_checkpoint(checkpoint_path)
model.eval()

# 2. Define standard Gemma components to read the language output
# (You can pull the standard lm_head linear layer from a base Gemma-2B)
gemma_lm_head = load_base_gemma_lm_head() 
tokenizer = load_gemma_tokenizer()

# 3. Intercept during custom inference
with torch.no_grad():
    # Pass your custom image and task instruction prompt
    inputs = model.prepare_inputs(image=camera_frame, text="pick up the red block")
    
    # Run the model up until the VLM backbone output
    hidden_states = model.vlm_backbone(inputs.image_tokens, inputs.text_tokens)
    
    # 4. Map the fine-tuned hidden states back to vocabulary logits
    logits = gemma_lm_head(hidden_states)
    predicted_token_ids = torch.argmax(logits, dim=-1)
    
    # 5. Decode into readable text
    readable_thought = tokenizer.decode(predicted_token_ids)
    print(f"What the fine-tuned LLM sees/thinks: {readable_thought}")
