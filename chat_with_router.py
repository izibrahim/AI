from langchain_community.llms import Ollama
import logging
import re
from netmiko import ConnectHandler

# Initialize the Llama3 model
llm = Ollama(model="llama3", base_url="http://localhost:11434", verbose=True)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Function to extract Python code from text
def extract_python_code(text):
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    return '\n'.join(code_blocks).strip()

# Function to send prompt to Llama3 and get the response
def send_prompt_to_llama3(prompt):
    response = llm.generate(prompts=[prompt])
    
    # Aggregate the response chunks
    aggregated_response = ""
    for chunk in response.generations[0]:
        aggregated_response += chunk.text

    return aggregated_response.strip()

# Define a function to execute Python code and return the output
def execute_python_code(python_code):
    local_vars = {}
    exec(python_code, {}, local_vars)
    return local_vars.get('output', 'No output generated.')

# Main logic
def main():
    # First prompt to generate the Python script
    initial_prompt = """
    write python script to ssh to the below node using netmiko connecthandler and get the show version use the below code as 

	from netmiko import ConnectHandler

# Define the device details
device = {
    'device_type': 'cisco_xr',
    'host': 'sandbox-iosxr-1.cisco.com',
    'username': 'admin',
    'password': 'C1sco12345',
    'port': 22,  # optional, default is 22
     'verbose': True,
}

# Connect to the device
connection = ConnectHandler(**device)

DO NOT SHARE ANY EXPLANATION
    """
    generated_code_response = send_prompt_to_llama3(initial_prompt)
    logging.info(f"Generated code response:\n{generated_code_response}")

    # Extract Python code from the response
    generated_code = extract_python_code(generated_code_response)
    logging.info(f"Extracted Python code:\n{generated_code}")

    # Execute the generated Python code
    try:
        output = execute_python_code(generated_code)
        logging.info(f"Execution output:\n{output}")

        # Send the output back to Llama3 for explanation
        explanation_prompt = f"Explain the following output from the code execution:\n{output}"
        explanation_response = send_prompt_to_llama3(explanation_prompt)
        logging.info(f"Explanation response:\n{explanation_response}")

    except Exception as e:
        logging.error(f"Error executing generated code: {e}")

if __name__ == "__main__":
    main()
