import openai
import os
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API key
openai.api_key = api_key

def generate_response(prompt):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",  # Change to 'gpt-3.5-turbo' or 'gpt-4' for other models
            prompt=prompt,
            max_tokens=150  # Limit the response length
        )
        # Extract and return the generated text
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {e}"

# Example: Using a prompt
prompt = "Explain the concept of black holes in simple terms."
response = generate_response(prompt)
print("Response:", response)
