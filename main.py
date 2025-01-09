import time
import onnxruntime_genai as engine


model_path = "phi3.5/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
start = time.time()
print(f"Loading model from {model_path}")
model = engine.Model(str(model_path))
print(f"Model loaded in {time.time() - start:.2f} seconds")


model = engine.Model(f'{model_path}')
tokenizer = engine.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the maximum number of tokens to generate
# The full example includes additional parameters for more complex configurations
search_options = {'max_length': 2048}

# Template for formatting prompt messages in the script
chat_tpl = '<|user|>\n{input}<|end|>\n<|assistant|>'

print("Let's chat!\n")

# Chat loop. Exit with Ctrl+C
try:
    while True:
        text = input("> User: ")
        if not text:
            print("Please, answer something")
            continue

        # Populate the chat template with user input
        prompt = f'{chat_tpl.format(input=text)}'

        input_tokens = tokenizer.encode(prompt)

        gen_params = engine.GeneratorParams(model)
        gen_params.set_search_options(**search_options)
        gen_params.input_ids = input_tokens
        generator = engine.Generator(model, gen_params)

        print("\n> Assistant: ", end='', flush=True)

        try:
            # Loop to generate and display answer tokens
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                next_token = generator.get_next_tokens()[0]
                print(tokenizer_stream.decode(next_token), end='', flush=True)
            print('\n')
        except KeyboardInterrupt:
            print("\nCtrl+C pressed, stopping chat inference")
except KeyboardInterrupt:
    pass

if "generator" in locals():
    # Delete the generator
    del generator