# Granite 3.2 Majority Voting

Granite 3.2 provides lots of cool capabilities like [chain-of-thought (CoT)](https://www.ibm.com/think/topics/chain-of-thoughts) reasoning, document understanding etc. In this example, we will investigate how to use majority voting to find the best possible answer from multiple answers (multi-completion) per prompt. The example will first cover how to get mutliple answers from a model in a vanilla fashion and then how to improve on getting the best answer using the [Granite IO Processing](https://github.com/ibm-granite/granite-io) library.

## Majority Voting Example

Large Language Models (LLMs) can perform a lot of tasks really well but maths sometimes is not one of them. Yes, you could get a specially trained model for maths. If however you don't have such a model, wouldn't it be great to be able to use a standard model. If we know that the model we are using may not always provide the correct answer to some maths problem, we then need to find a way where we can guide it. One such way is to ask the model to run the prompt a number of times and this is called multi-completions. Instead of getting one answer from the model we will get multiple. We can then compare to see which answer looks best or most correct.

To try this concept, lets ask `Granite 3.2 8b` model the following question `What is 234651 + 13425?` and then work towards how to to get the best/most accurate answer. 

### Prompt the model

First off, lets start with a simple sample code snippet below to ask the Granite 3.2 model the maths question
```py
import openai

# This is setup to use a local Ollam server for inferece with default.
# Ollama server does not support multi-completions i.e. "n=20"
# Thererfore, to show multiple output for the prompt, you need to use a runtime
# that does e.g, vLLM. The seeings will then need to be updated accordingly.
api_key = "ollama"
base_url = "http://localhost:11434/v1" # Ollama server hosting model
default_model = "granite3.2:8b"

openai_client = openai.OpenAI(
    base_url=base_url, api_key=api_key
)

prompt = "What is 234651 + 13425?\nAnswer with just a number please."

result = openai_client.completions.create(
    model=default_model,
    prompt=prompt,
    n=20,
    temperature=0.6,
    max_tokens=1024,
)
results = []
print("\n\n")
num_result = 1
for choice in result.choices:
    print(f"**Result {num_result}**: {choice.text}")
    num_result += 1
```

The model will generate output similar to the following (this can potentially vary a lot per run):

```shell
**Result 1**: 

468906
**Result 2**: 

369176
**Result 3**: 

248076
**Result 4**: 

368906
**Result 5**: 

468906
**Result 6**: 

358976
**Result 7**: 

468906
**Result 8**: 

368876
**Result 9**: 

368906
**Result 10**: 

369176
**Result 11**: 

368876
**Result 12**: 

468906
**Result 13**: 

568906
**Result 14**: 

468906
**Result 15**: 

568906
**Result 16**: 

568906
**Result 17**: 

368876
**Result 18**: 

368976
**Result 19**: 

369176
**Result 20**: 

369176
```

The problem with this is that the:

- Outputs vary from run to run
- After each run, you need to parse through 20 odd answers
- The correct answer isn't always in the majority (above output is poor)

Follow onto the next example on how to prompt the model for better answers.

### Prompt model to improve the answers

This time when we ask the question `What is 234651 + 13425?\nAnswer with just a number please.`, lets also try and improve the answers returned:

```py
import openai

# This is setup to use a local Ollam server for inferece with default.
# Ollama server does not support multi-completions i.e. "n=20"
# Thererfore, to show multiple output for the prompt, you need to use a runtime
# that does e.g, vLLM. The seeings will then need to be updated accordingly.
api_key = "ollama"
base_url = "http://localhost:11434/v1" # Ollama server hosting model
default_model = "granite3.2:8b"

openai_client = openai.OpenAI(
    base_url=base_url, api_key=api_key
)

#prompt = "What is 234651 + 13425?\nAnswer with just a number please."
prompt = """<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: April 02, 2025.
You are Granite, developed by IBM. You are a helpful AI assistant.
Respond to every user query in a comprehensive and detailed way. The response should summarize the thought process. Write your thoughts after 'Here is my thought process:' and write your response after 'Here is my response:' and within '<answer>' and  '</answer>.' Only give me a single thought and a single final response, nothing else.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>What is 234651 + 13425?
Answer with just a number please.<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""

result = openai_client.completions.create(
    model=default_model,
    prompt=prompt,
    n=20,
    temperature=0.6,
    max_tokens=1024,
)
results = []
print("\n\n")
num_result = 1
for choice in result.choices:
    print(f"**Result {num_result}**: {choice.text}")
    num_result += 1
```

Now the response from the model contains more context on the answer :

```shell
**Result 1**: Here is my thought process:
The task is to add two numbers, 234651 and 13425, and provide the sum as a standalone numerical answer.

Here is my response:

<answer>248076</answer>
**Result 2**: Here is my thought process:
The task is to calculate the sum of two numbers, 234651 and 13425. I need to add these numbers together.

Here is my response:

<answer>248076</answer>
**Result 3**: Here is my thought process:
The task is to add two numbers, 234651 and 13425, and provide the sum as a standalone number.

Here is my response:

<answer>248076</answer>
**Result 4**: Here is my thought process:
To find the sum of 234651 and 13425, we need to add these two numbers together.

Here is my response:

<answer>248076</answer>
**Result 5**: Here is my thought process:
The task requires me to calculate the sum of two numbers, 234651 and 13425. I will add these two numbers together to get the result.

Here is my response:

<answer>248076</answer>
**Result 6**: Here is my thought process:
To find the sum of 234651 and 13425, we need to add these two numbers together.

Here is my response:

<answer>348076</answer>
**Result 7**: Here is my thought process:
The task is to add two numbers, 234651 and 13425, and provide the result.

Here is my response:

<answer>248076</answer>
**Result 8**: Here is my thought process:
To find the sum, we need to add 234651 and 13425 together.

Here is my response:

<answer>248076</answer>
**Result 9**: Here is my thought process:
To find the answer, we need to add the two numbers together.

Here is my response:

1010266

</answer>
**Result 10**: Here is my thought process:
The task is to calculate the sum of two numbers, 234651 and 13425. I need to perform addition to find the result.

Here is my response:

<answer>248076</answer>
**Result 11**: Here is my thought process:
To find the sum, we need to add 234651 and 13425.

Here is my response:

<answer>248076</answer>
**Result 12**: Here is my thought process:
The question is asking for the sum of two numbers, 234651 and 13425. To find the answer, we need to add these two numbers together.

Here is my response:

<answer>248076</answer>
**Result 13**: Here is my thought process:
To find the sum of 234651 and 13425, we need to add these two numbers together.

Here is my response:

<answer>248076</answer>
**Result 14**: Here is my thought process:
To find the sum of 234651 and 13425, we need to add these two numbers together.

Here is my response:

<answer>248076</answer>
**Result 15**: Here is my thought process:
To find the sum of 234651 and 13425, we need to add these two numbers together.

Here is my response:

<answer>248076</answer>
**Result 16**: Here is my thought process:
To find the sum of 234651 and 13425, we need to add these two numbers together.

Here is my response:

<answer>248076</answer>
**Result 17**: Here is my thought process:
The question is asking to add two numbers: 234651 and 13425. The task is to perform this arithmetic operation and provide the final sum.

Here is my response:

<answer>248076</answer>
**Result 18**: Here is my thought process:
To find the sum of 234651 and 13425, we need to add these two numbers together.

Here is my response:

<answer>248076</answer>
**Result 19**: Here is my thought process:
To find the sum, we need to add 234651 and 13425.

Here is my response:

<answer>248076</answer>
**Result 20**: Here is my thought process:
The task is to calculate the sum of two numbers, 234651 and 13425. I need to perform addition to find the result.

Here is my response:

<answer>248076</answer>
```

The updated prompt shows a marked improvement on the previous prompt with just a few hallucinations (incorrect answers) this time.

However, in the code snippet above you can see that we had to make a substantial and intricate change to the prompt for the model to provide better answers. There are 20 answers also that we need to parse through to understand what is the best answer. This is not something we want to have to do every time to get the correct answer from the model.  Wouldn't it be great if we could be abstracted from this and only require to provide the question or prompt. This is where the [Grabite IO Processing](https://github.com/ibm-granite/granite-io) library comes into play. Follow onto the next section to see how it can simplify your task.

### Use granite-io to help with choosing the correct answer

Here is how the code snippet looks when using the `granite-io` library to do the heavy lifting:

```
"""
This example show how to infer or call a model using the framework and an Ollama
backend to serve the model.

It uses majority voting to decide on best answer to use from number of model sample
outputs.
"""

# Local
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

# By default the backend is an Ollama server running locally using default url. Ollama
# howevere does not support multiple completions per prompt i.e. "n" param below.
# You will need to use a backend that supports multiple completions to fully see
# majority voting happening. Backends like vLLM support multiple completions. You will
# need to update url and key when connecting to another backend.
model_name = "granite3.2:8b"
# openai_base_url = ""
# openai_api_key = ""
io_processor = make_io_processor(
    model_name,
    backend=make_backend(
        "openai",
        {
            "model_name": model_name,
            # "openai_base_url": openai_base_url,
            # "openai_api_key": openai_api_key,
        },
    ),
)
question = "What is 234651 + 13425?\nAnswer with just a number please."
messages = [UserMessage(content=question)]
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(
        messages=messages,
        majority_voting=True,
        generate_inputs={"n": 20, "temperature": 0.6, "max_tokens": 1024},
    )
)
print("------ WITH MAJORITY VOTING ------")
# There will be only 1 output because majority voting is performed
# on all completions results
num_result = 1
for result in outputs.results:
    print(f"**Result {num_result}**: {result.next_message.content}")
    num_result += 1
```

Gone is the long unwielding prompt to be replaced by a more easily maintable and understandable way to:
- Set the prompt
- Request majority voting (`majority_voting=true`)

The output returned from the model is now as follows:

```shell
------ WITH MAJORITY VOTING ------
**Result 1**: 248076
```

The output now just contains the 1 answer which is the answer as selected by majority voting on the 20 answers as returned by the model. It is also proving to be the correct answer. That simplifies my job! 

## Conclusion

- You can prompt a Granite 3.2 model asking to return answers on a task that it is not best trained for
- However, this requires creating quite an unwielding prompt which is error prone and hard to extend
- It also requires parsing of different outputs to arrive at best/correct answer
- [Granite IO Processing](https://github.com/ibm-granite/granite-io) provides an abstracted and easy to use the library where you can specify the:
  - Input to the model you want. For this example, basic prompt
  - Output from the model. For this example, you perform majority voting on multiple answers, and return the majority answer


 
