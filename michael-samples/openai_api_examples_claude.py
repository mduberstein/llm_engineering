import openai
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI client
# Make sure to set your API key as an environment variable: OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Example 1: Basic chat completion
def basic_chat_example():
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        print("Basic Chat Response:")
        print(response.choices[0].message.content)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in basic chat: {e}")

# Example 2: Streaming response
def streaming_chat_example():
    try:
        print("Streaming Response:")
        stream = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Tell me a short story about a robot."}
            ],
            stream=True,
            max_tokens=200
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"Error in streaming chat: {e}")

# Example 3: Chat with conversation history
def conversation_example():
    try:
        messages = [
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "What about 3 + 5?"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        
        print("Conversation Response:")
        print(response.choices[0].message.content)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in conversation: {e}")

# Example 4: Using function calling
def function_calling_example():
    try:
        # Define a function that the model can call
        functions = [
            {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "What's the weather like in New York?"}
            ],
            functions=functions,
            function_call="auto"
        )
        
        print("Function Calling Response:")
        message = response.choices[0].message
        if message.function_call:
            print(f"Function called: {message.function_call.name}")
            print(f"Arguments: {message.function_call.arguments}")
        else:
            print(message.content)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in function calling: {e}")

# Example 5: Text completion (legacy endpoint)
def text_completion_example():
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt="Complete this sentence: The future of AI is",
            max_tokens=50,
            temperature=0.8
        )
        
        print("Text Completion Response:")
        print(response.choices[0].text.strip())
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in text completion: {e}")

# Example 6: Image generation with DALL-E
def image_generation_example():
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt="A futuristic city with flying cars at sunset",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        print("Image Generation Response:")
        print(f"Image URL: {response.data[0].url}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in image generation: {e}")

# Example 7: Embeddings
def embeddings_example():
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a sample text to embed."
        )
        
        print("Embeddings Response:")
        print(f"Embedding dimensions: {len(response.data[0].embedding)}")
        print(f"First 5 values: {response.data[0].embedding[:5]}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error in embeddings: {e}")

# Example 8: Error handling and rate limiting
def robust_api_call():
    import time
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "Explain quantum computing in simple terms."}
                ],
                max_tokens=200
            )
            
            print("Robust API Call Response:")
            print(response.choices[0].message.content)
            break
            
        except openai.RateLimitError:
            print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
            
        except openai.APIError as e:
            print(f"API error: {e}")
            break
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Max retries exceeded.")

# Main execution
if __name__ == "__main__":
    print("OpenAI API Examples")
    print("=" * 50)
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable")
        exit(1)
    
    # Run examples
    basic_chat_example()
    streaming_chat_example()
    conversation_example()
    function_calling_example()
    text_completion_example()
    image_generation_example()
    embeddings_example()
    robust_api_call()
