import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load the HUGGINGFACEHUB_API_TOKEN from your .env file
load_dotenv()

# Get the token
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    print("ERROR: Hugging Face token not found in .env file!")
else:
    print("Token found. Attempting to connect to Hugging Face...")
    try:
        # Initialize the client
        client = InferenceClient(token=hf_token)

        # Try to call the API with a simple prompt
        response = client.chat_completion(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            model="mistralai/Mistral-7B-Instruct-v0.2",  # The model we are testing
            max_tokens=50,
        )

        print("\nSUCCESS! The connection works.")
        print("Response from model:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"\nFAILURE! The connection failed.")
        print(f"The error was: {e}")