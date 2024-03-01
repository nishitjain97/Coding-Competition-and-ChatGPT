"""
File with utilities.

Nishit Jain, November 2023
Nishit Jain, March 2024

References:
    - Stanley Bak, March 2023
"""
from openai import OpenAI
import requests
import tiktoken

###
# create_prompt
###

def create_prompt(
        description: dict,
        prompt_template: str,
        outfile: str=None
)->str:
    """
        Function to create prompt using given template.

        Arguments:
            - description (dict): A dictionary. Key specifies special string in the prompt template and value specifies item to replace the special string with.
            - prompt_template (str): A string that gives the template for the prompt.
            - outfile (str): Location of output file.

        Returns:
            (str) Prompt as a string.
    """
    for key, value in description.items():
        prompt_template = prompt_template.replace(key, value)

    if outfile is not None:
        with open(outfile, 'w') as f:
            f.writelines(prompt_template)

    return prompt_template

###
# num_tokens_from_messages
###
def num_tokens_from_messages(
        messages: list,
        model: str
)->int:
    """
        Function to count number of tokens.

        Arguments:
            - messages (list): List of messages to GPT
            - model (str): Name of model being used.

        Returns:
           (int) Number of tokens in the message 
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    if model == 'gpt-3.5-turbo':
        return num_tokens_from_messages(
            messages,
            model="gpt-3.5-turbo-0301"
        )
    elif model == "gpt-4":
        return num_tokens_from_messages(
            messages,
            model="gpt-4-0314"
        )
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""
                num_tokens_from_message() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.
            """
        )
    num_tokens = 0

    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))

            if key == "name":
                num_tokens += tokens_per_name
    
    num_tokens += 3

    return num_tokens


###
# ask_chatgpt
###
def ask_chatgpt(
        client: 'OpenAI',
        prompt: str,
        model: str,
        temperature: float = 0,
        stdout: bool = True,
        n: int = 1
)->'dict':
    """
        Function to send prompt to ChatGPT.

        Arguments:
            - client (OpenAI): OpenAI client
            - prompt (str): Prompt
            - model (str): Model name
            - temperature (float): To control randomness
            - stdout (bool): To print to standard output
            - n (int): Number of alternative responses to the prompt

        Returns:
            dict: Full response and code blocks for each of n alternative responses
    """
    if stdout:
        print(f"Asking ChatGPT prompt:\n{prompt}")

    try:
        responses = ask_chatgpt_streaming(
            client=client,
            prompt=prompt,
            model=model,
            temperature=temperature,
            stdout=stdout,
            n = n
        )
    except(
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
        AssertionError
    ) as e:
        print(f"Error during ChatGPT generation: {e}")

    return responses


###
# ask_chatgpt_streaming
###
def ask_chatgpt_streaming(
        client: 'OpenAI',
        prompt: str,
        model: str,
        temperature: float = 0,
        stdout: bool = True,
        n: int = 1
)->'dict':
    """
        Function to query chatgpt with prompt and get the full response and code blocks.

        Arguments:
            - client (OpenAI): OpenAI client
            - prompt (str): String with prompt for GPT
            - model (str): String with model name
            - temperature (float): To control randomness. 0 means no randomness (consistency in responses)
            - stdout (bool): Print messages on standard output
            - n (int): Number of alternative responses to the prompt

        Returns:
            dict: Full response and code blocks for each of n alternative responses
    """
    system_message = {
        "role": "system",
        "content": "You are an expert Python programmer who can code anything."
    }

    user_message = {
        "role": "user",
        "content": prompt
    }

    assert model in ["gpt-3.5-turbo", "gpt-4"]
    messages = [system_message, user_message]

    MAX_TOKENS = 4096 if model == "gpt-3.5-turbo" else 8192

    total_message_tokens = num_tokens_from_messages(
        messages,
        model
    )
    max_response_tokens = min(MAX_TOKENS // 2 - 1, MAX_TOKENS - total_message_tokens)

    if stdout:
        print(f"Sending request to {model} (max_response_tokens = {max_response_tokens})", flush=True)
        
    response_iterator = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_response_tokens,
        stream=True,
        n=n
    )
    
    # Keep track of the n different streaming responses
    responses = {}
    for i in range(n):
        responses[i] = {'chunks': []}

    for chunk in response_iterator:
        responses[chunk.choices[0].index]['chunks'].append(chunk)

    def parse_response(chunks: list) -> tuple:
        """
            Function to parse chunks and return tuple with complete response and code blocks.

            Arguments:
                - chunks (list): List of chunks from GPT API

            Returns:
                (tuple): A two-tuple of complete response and code blocks.
        """
        finish_reason = None
        cur_content = ""
        started = False
        full_response = ""

        code_blocks = []
        cur_code_block = ""

        for chunk in chunks:
            finish_reason = chunk.choices[0].finish_reason
            content = chunk.choices[0].delta.content

            if content is not None:
                cur_content += content
                full_response += content

                if not started:
                    if stdout:
                        print(content, end="", flush=True)

                    code_prefixes = [
                        "```python\n",
                        "```\n",
                        "'''\n"
                    ]

                    for code_prefix in code_prefixes:
                        index = cur_content.find(code_prefix)

                        if index != -1:
                            break

                    if index != -1:
                        cur_content = cur_content[index + len(code_prefix):]
                        started = True
                        cur_code_block += cur_content
                        cur_content = ""
                else:
                    code_suffixes = [
                        "```",
                        "'''"
                    ]

                    for code_suffix in code_suffixes:
                        index = cur_content.find(code_suffix)

                        if index != -1:
                            break
                    
                    if index != -1:
                        started = False
                        cur_content = cur_content[:index]

                    last_char_is_backtick = cur_content.rfind("`") == len(cur_content) - 1 or cur_content.rfind("'") == len(cur_content) - 1

                    if not last_char_is_backtick:
                        cur_code_block += cur_content
                        cur_content = ""
                    
                    if not started:
                        code_blocks.append(cur_code_block)
                        cur_code_block = ""

        if finish_reason != "stop":
            print(f"Finish reason was not 'stop': {finish_reason}")

        return full_response, code_blocks
    
    for i in range(n):
        full_response, code_blocks = parse_response(responses[i]['chunks'])
        responses[i]['full_response'] = full_response
        responses[i]['code_blocks'] = code_blocks
        responses[i]['chunks'] = None

    return responses