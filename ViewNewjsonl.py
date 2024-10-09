import json
def view_conversations(filename):
        with open(filename, 'r', encoding='utf8') as f:
            # Read all lines from the JSONL file
            lines = f.readlines()

            current_system_prompt = None
            conversation_entries = []
            count = 0
            for line in lines:
                # Parse the JSON entry
                conversation_entry = json.loads(line)
                
                if current_system_prompt != conversation_entry["system_prompt"]:
                    # If this is not the first conversation, print the previous one
                    if conversation_entries:
                        print(f"Conversation:{count}")
                        # Print the system prompt only once
                        #print(f"\nSystem Prompt: {current_system_prompt}")
                        for user_input, model_response in conversation_entries:
                            #print(f"User: {user_input}")
                            print(f"Customs Agent: {model_response}")
                        print("\n")
                    count +=1
                    current_system_prompt = conversation_entry["system_prompt"]
                    conversation_entries = []

                user_input = conversation_entry["user_input"]
                model_response = conversation_entry["model_response"]
                conversation_entries.append((user_input, model_response))

            if conversation_entries:
                #print(f"\nSystem Prompt: {current_system_prompt}\n\n")
                for user_input, model_response in conversation_entries:
                    #print(f"User: {user_input}")
                    print(f"Customs Agent: {model_response}")
                print("\n")  # Blank line for separation

            # Wait for user input to exit
            input("Press Enter to exit...")

# Usage
if __name__ == "__main__":
    output_filename = "modelresponsesv1.jsonl" 
    view_conversations(output_filename)

