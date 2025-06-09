from vllm import LLM, SamplingParams

def run_vllm_with_tensor_parallelism():
    '''
    Run a simple test with vLLM using tensor parallelism.
    '''

    model_name = "Qwen/Qwen3-8B"

    tp_size = 2  # Set tensor parallelism size

    prompts = [
        "The future of AI is",
        "The impact of quantum computing on technology is",
        "What are the benifits of vLLM?"
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        stop=["\n\n"] # Stop generation on double newlines
    )

    print(f"Initializing LLM engine for model: {model_name} with tensor parallelism size: {tp_size}")
    print("This may take a while...")

    try:
        llm = LLM(model=model_name, tensor_parallel_size=tp_size)
        print("LLM engine initialized successfully.")
        print("Generating responses...")

        outputs = llm.generate(prompts, sampling_params=sampling_params)

        for output in outputs:
            prompt = output.prompt
            response = output.outputs[0].text
            print(f"Prompt: {prompt}\nResponse: {response}\n")
    except Exception as e:
        print(f"An error occurred while running vLLM: {e}")
        
if __name__ == "__main__":
    run_vllm_with_tensor_parallelism()
