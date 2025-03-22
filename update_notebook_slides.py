import json

# Read the notebook
with open('notebooks/01_training_simple_gpt.ipynb', 'r') as f:
    notebook = json.load(f)

# Update cell metadata for slides
slide_metadata = {
    0: {"slideshow": {"slide_type": "slide"}},  # Title slide
    1: {"slideshow": {"slide_type": "slide"}},  # Imports
    2: {"slideshow": {"slide_type": "slide"}},  # Model Configuration markdown
    3: {"slideshow": {"slide_type": "subslide"}},  # Model Configuration code
    4: {"slideshow": {"slide_type": "slide"}},  # Data Loading markdown
    5: {"slideshow": {"slide_type": "subslide"}},  # Data Loading code
    6: {"slideshow": {"slide_type": "slide"}},  # Model Initialization markdown
    7: {"slideshow": {"slide_type": "subslide"}},  # Model Initialization code
    8: {"slideshow": {"slide_type": "slide"}},  # Training Loop markdown
    9: {"slideshow": {"slide_type": "subslide"}},  # Training Loop code
}

# Update metadata for each cell
for idx, metadata in slide_metadata.items():
    if idx < len(notebook['cells']):
        notebook['cells'][idx]['metadata'].update(metadata)

# Save the updated notebook
with open('notebooks/01_training_simple_gpt.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Successfully updated notebook with slide metadata") 