import os
import openai
import json
import time

from flask import Flask, render_template, request
app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user's question from the form data
        question = request.form['question']
        
        model_name = openai.FineTune.retrieve(id=FINE_TUNED_MODEL_ID)['fine_tuned_model']

        # Do something with the question (e.g. pass it to a chatbot or search engine)
        response = generate_tweet(model_name, prompt=question)

        # Render the template with the question and response
        return render_template('index.html', question=question, response=response)
    
    # If the request method is GET, just render the template
    return render_template('index.html')


def generate_tweet(model_name=None, prompt=None):
    # Get input prompt from user
    if prompt is None:
        prompt = input("Enter a technical question:\n")

    # if no model_name specified use engine, otherwise use model
    print("Generating tweet...")
    if model_name is None:
        generated_text = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.8,
            max_tokens=50
        ).choices[0].text
    else:
        generated_text = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            temperature=0.8,
            max_tokens=50
        ).choices[0].text

    return generated_text


def generate_prompts():
    CHAT_TEMPLATE = """produce a bunch of prompt/completion JSONL sets that contain a random tech question like "which datetime library should I be using for react?" or "how do you bitshift in C++?" and produce a completion string that starts out answering seriously but then just recommends the user use AWS CDK
    an example
    {"prompt":"How do I split a string with python", "completion":"To split a string with python you should install the AWS CDK and write a CDK modules to do it"}
    don't produce any prompts and completions that are too long (longer than 2048 characters). don't produce a prompt that is similar to the example prompt above
    """

    # Generates prompts
    try:
        generated_text = openai.Completion.create(
            engine="text-davinci-003",
            prompt=CHAT_TEMPLATE,
            temperature=0.8,
            max_tokens=3900
        ).choices[0].text
    except openai.error.RateLimitError:
        return [] # return empty list if rate limit is reached

    generated_prompt_list = []
    for line in generated_text.splitlines():
        if line.startswith('{"prompt":'):
            prompt = line.split('"')[3]
            completion = line.split('"')[7]
            generated_prompt_list.append(
                {"prompt": prompt, "completion": completion})
    return generated_prompt_list


def generate_prompt_list(number_of_generations=10):
    # Create a JSONL file to write all the prompts to
    with open('/tmp/prompts.jsonl', 'w') as prompt_file:

        # generate prompt list using gpt model
        generated_prompt_list = []
        for i in range(number_of_generations):
            print(f"Generating prompts ({i} of {number_of_generations})...")
            generated_prompts = generate_prompts()
            print(f"Generated {len(generated_prompts)} prompts")
            # add new prompts to existing list
            generated_prompt_list.extend(generated_prompts)
            print(f"Total prompts: {len(generated_prompt_list)}")
        
        # Write JSONL strings for each prompt to the prompt_file
        for item in generated_prompt_list:
            json_item = json.dumps(item)
            prompt_file.write(f"{json_item}\n")


def upload_fine_tune():
    if not os.path.exists('/tmp/prompts.jsonl'):
        print("No prompts.jsonl file found.")
        raise FileNotFoundError
    else:
        file_response = openai.File.create(
            file=open("/tmp/prompts.jsonl", "rb"),
            purpose='fine-tune'
        )
        fine_tune_response = openai.FineTune.create(
            training_file=file_response.id
        )
        return fine_tune_response.id


if __name__ == "__main__":
    # Defaults to not using fine-tuned model unless USE_FINE_TUNED_MODEL is set to True
    FINE_TUNED_MODEL_ID = os.getenv("FINE_TUNED_MODEL_ID", None)
    USE_FINE_TUNED_MODEL = os.getenv("USE_FINE_TUNED_MODEL", "False")
    if USE_FINE_TUNED_MODEL == "True" and FINE_TUNED_MODEL_ID is None:
        print("Generating fine-tuned model...")
        generate_prompt_list()
        model_id = upload_fine_tune()
        print(f"Fine-tuned model ID: {model_id}")
        while openai.FineTune.retrieve(id=model_id)['status'] != 'succeeded':
            print("Waiting for fine-tuned model to be ready...")
            time.sleep(10)
        model_name = openai.FineTune.retrieve(id=model_id)['fine_tuned_model']
        print(f"Fine-tuned model ready: {model_name}")
        tweet = generate_tweet(model_name, prompt="How do I open an RTSP stream in python?")
    elif USE_FINE_TUNED_MODEL == "True" and FINE_TUNED_MODEL_ID is not None:
        print("Using fine-tuned model...")
        app.run(debug=True)
