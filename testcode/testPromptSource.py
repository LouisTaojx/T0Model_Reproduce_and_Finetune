from promptsource.templates import DatasetTemplates

# coed for testing promptsource templates
# Load templates for the SuperGLUE CB task
templates = DatasetTemplates("super_glue", "cb")

# List available prompts
print("1.Available prompts:", templates.all_template_names)

# Use a specific prompt
template = templates["does this imply"]
example = {
    "premise": "The sky is blue.",
    "hypothesis": "It is daytime."
}
input_text = template.apply(example)[0]
print("2.Generated prompt:", input_text)
