# from unsloth import FastLanguageModel                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                   
# print("Loading merged model...")                                                                                                                                                                             
# model, tokenizer = FastLanguageModel.from_pretrained(                                                                                                                                                        
#     model_name = "./agri-gemma-4-e4b-it",                                                                                                                                                                       
#     max_seq_length = 8192,                                                                                                                                                                                   
#     dtype = None,                                                                                                                                                                                            
#     load_in_4bit = False,                                                                                                                                                                                    
# )

# print("Exporting model to GGUF format...")
# model.save_pretrained_gguf("./gemma-cpt-gguf", tokenizer, quantization_method = "f16")
# print("GGUF export complete!")


import os                                                                                                                                                                                                    
import json                                                                                                                                                                                                  
                                                                                                                                                                                                             
MODEL_DIR = "./outputs/sft/gemma4_e2b_acid_soils_sft_merged"                                                                                                                                                                          
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor_config.json")                                                                                                                                      
                                                                                                                                                                                                             
# Patch missing image processor fields for GGUF compatibility                                                                                                                                                
if os.path.exists(preprocessor_path):                                                                                                                                                                        
    with open(preprocessor_path, "r") as f:                                                                                                                                                                  
        preprocessor_config = json.load(f)                                                                                                                                                                   
else:                                                                                                                                                                                                        
    preprocessor_config = {}                                                                                                                                                                                 
                                                                                                                                                                                                             
# Inject Gemma 4 SigLIP defaults if missing                                                                                                                                                                  
defaults = {                                                                                                                                                                                                 
    "image_mean": [0.5, 0.5, 0.5],                                                                                                                                                                           
    "image_std": [0.5, 0.5, 0.5],                                                                                                                                                                            
    "do_normalize": True,                                                                                                                                                                                    
    "do_resize": True,                                                                                                                                                                                       
    "size": {"height": 896, "width": 896},                                                                                                                                                                   
    "resample": 2,                                                                                                                                                                                           
}                                                                                                                                                                                                            
patched = False                                                                                                                                                                                              
for key, val in defaults.items():                                                                                                                                                                            
    if key not in preprocessor_config:                                                                                                                                                                       
        preprocessor_config[key] = val                                                                                                                                                                       
        patched = True                                                                                                                                                                                       
                                                                                                                                                                                                             
if patched:                                                                                                                                                                                                  
    with open(preprocessor_path, "w") as f:                                                                                                                                                                  
        json.dump(preprocessor_config, f, indent=2)                                                                                                                                                          
    print(f"✅ Patched preprocessor_config.json with missing image processor fields.")                                                                                                                       
                                                                                                                                                                                                             
# --- Then proceed with your existing code ---                                                                                                                                                               
from unsloth import FastLanguageModel                                                                                                                                                                        
                                                                                                                                                                                                             
print("Loading merged model...")                                                                                                                                                                             
model, tokenizer = FastLanguageModel.from_pretrained(                                                                                                                                                        
    model_name = MODEL_DIR,                                                                                                                                                                                  
    max_seq_length = 8192,                                                                                                                                                                                   
    dtype = None,                                                                                                                                                                                            
    load_in_4bit = False,                                                                                                                                                                                    
)                                                                                                                                                                                                            
                                                                                                                                                                                                             
print("Exporting model to GGUF format...")                                                                                                                                                                   
model.save_pretrained_gguf("./outputs/sft/gguf_v2/", tokenizer, quantization_method = "bf16")                                                                                                                      
print("GGUF export complete! 🚀")
