# OpenAI Fine-tune shill

Train your fine-tuned model

```bash
export USE_FINE_TUNED_MODEL=True
python generate.py
```

Once trained, a fine-tuned model ID will be provided back to you. This can be used for future chat completition by setting the following variable and re-running the script

```bash
export USE_FINE_TUNED_MODEL=True
export FINE_TUNED_MODEL_ID=ft-xxxxxxxxxxxxxxxxxxxxxxxx
python generate.py
```
