from abc import ABC
import time
import onnxruntime_genai as engine


class Tokens(ABC):
    SYSTEM = None
    USER = None
    ASSISTANT = None
    END = None


class PhiTokens(Tokens):
    SYSTEM = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    END = "<|end|>"


class LlamaTokens(Tokens):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError("Llama tokens not implemented yet")


model_path = "phi3.5/cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4"
start = time.time()
print(f"Loading model from {model_path}")
model = engine.Model(str(model_path))
print(f"Model loaded in {time.time() - start:.2f} seconds")

model = engine.Model(f"{model_path}")
tokenizer = engine.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
tokens = PhiTokens()

# Set the maximum number of tokens to generate
# The full example includes additional parameters for more complex configurations
search_options = {"max_length": 2048}

print("Let's chat!\n")

# Initialize message history
message_history = []

# Chat loop. Exit with Ctrl+C
try:
    while True:
        text = input("> User: ")
        if not text:
            print("Please, ask something")
            continue

        # Add user input to message history
        message_history.append(f"{tokens.USER}\n{text}{tokens.END}")

        # Create the prompt from the message history
        prompt = "\n".join(message_history) + f"\n{tokens.SYSTEM}"

        input_tokens = tokenizer.encode(prompt)

        gen_params = engine.GeneratorParams(model)
        gen_params.set_search_options(**search_options)
        gen_params.input_ids = input_tokens
        generator = engine.Generator(model, gen_params)

        print("\n> Assistant: ", end="", flush=True)

        try:
            # Loop to generate and display answer tokens
            assistant_response = ""
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                next_token = generator.get_next_tokens()[0]
                decoded_token = tokenizer_stream.decode(next_token)
                assistant_response += decoded_token
                print(decoded_token, end="", flush=True)
            print("\n")

            # Add assistant response to message history
            message_history.append(f"{tokens.ASSISTANT}\n{assistant_response}{tokens.END}")
        except KeyboardInterrupt:
            print("\nCtrl+C pressed, stopping chat inference")
except KeyboardInterrupt:
    pass

# Clean up
if "generator" in locals():
    del generator
