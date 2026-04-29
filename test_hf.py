from openai import OpenAI

# Initialize the client pointing to Hugging Face
client = OpenAI(
    base_url="https://router.huggingface.co",
    api_key="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"
)

print("Sending request to Hugging Face...")

try:
    # Call the Llama 3 model hosted on Hugging Face
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "user", "content": "Explain what an IoT smart meter is in one sentence."}
        ],
        max_tokens=50
    )
    
    # Print the response from Llama 3
    print("\nResponse from Hugging Face (Llama 3):")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\nError: {e}")