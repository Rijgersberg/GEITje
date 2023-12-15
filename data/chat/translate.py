import json
import re
import time

from openai import OpenAI, BadRequestError

client = OpenAI()


def send_request(prompt):
    """GPT3.5 is by far the best translator I can find, but it is fickle.
    Somehow, using advanced AI has turned into repeatedly parsing broken output.
    In my experience, forcing json input and output is the best way to make sure
    GPT3.5 doesn't try to follow the prompts instead of translating them."""
    max_tokens = 2048
    for i in range(10):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a translation bot that translates JSON documents from English to Dutch. You output JSON only."""
                    },
                    {
                        "role": "user",
                        "content": "{\"content\": \"" + prompt + "\"}"
                    }
                ],
                temperature=1,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "json_object"},
            )
            translation = tuple(json.loads(response.choices[0].message.content).items())[0][1]
            if isinstance(translation, str):
                return translation
            else:
                print(f'Translation was not a single string. Trying again for try {i+2}')
                continue
        except BadRequestError as e:
            prompt_tokens = int(re.findall(r'\d+', e.message)[3])
            max_tokens = 4097 - prompt_tokens
            print(f'Requested too many tokens. Trying again with {max_tokens} tokens')
        except json.decoder.JSONDecodeError as e:
            print(f"JSONDecodeError {e}! Trying again for try {i+2}")
        except Exception as e:
            print(f"Exception {e}! Trying again for try {i+2} after 30 seconds")
            time.sleep(30)
    print('<TRANSLATION FAILED>')
    return '<TRANSLATION FAILED>'


def translate_conversation(conversation):
    return [{'role': turn['role'],
             'content': send_request(turn['content'])}
            for turn in conversation]
