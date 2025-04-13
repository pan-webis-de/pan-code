# Setup

For all provided OpenAI models, we utilize the following parameters and prompt:
1. `temperature=1.0`
2. `top_p=1.0`

```python
PROMPT = """
Your task is text style transfer. You rewrite the text into non-toxic language. You must
match the target style and preserve the original meaning as much as possible. You
cannot hallucinate or add anything outside the original input text. You should not include
the input text in the response. You should only generate the target text.
You should get rid of all toxic or impolite words or replace them.
You must respond on the original language of a given input text.
Please follow all instructions.
## Toxic input text:
{input_text}
## Detoxified text:
"""
```