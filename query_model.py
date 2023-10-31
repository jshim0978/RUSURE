from transformers import pipeline
from huggingface_hub import login



def main():
    # Load the model
    login()
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

    while True:
        # Take user input
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Generate response
        response = generator(user_input, max_length=50, num_return_sequences=1)

        # Output the response
        print(response[0]['generated_text'])

if __name__ == "__main__":
    main()
