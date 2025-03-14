# Granite 3.2 RAG

Granite 3.2 provided new capability like RAG. <TODO: add to>

## RAG Example

I am very proud of my sports team Limerick in the Irish Gaelic game of Hurling. It has helped that they have been the most success ful team over the last 10 years (2014-2024). In that period they have won:
- 5 All-Ireland finals (think super bowl)
- 4 in a row (2020-2023), only 2 other team to do this
- 6 Munster Titles
- 3 national Leagues
In other words the most successful in this period.

I decided then that it would be nice to ask `Granite 3.2 8b` model to confirm this with the following question `What team is the most successful hurling team in the last 10 years?"

### Prompt the model

Lets start with a simple sample code snippet below :

```py
import openai

api_key = "ollama"
base_url = "http://localhost:11434/v1" # Ollama server hosting model
default_model = "granite3.2:8b"

openai_client = openai.OpenAI(
    base_url=base_url, api_key=api_key
)

prompt = "What team is the most successful hurling team in the last 10 years?"

result = openai_client.completions.create(
    model=default_model,
    prompt=prompt,
    temperature=0.0,
)
results = []
print("\n\n")
for choice in result.choices:
    print(f"{choice.text}\n")
```

The model to my horror output the following:

```shell
As of my knowledge up to April 2023, the most successful hurling team over the past decade (2011-2021) has been Kilkenny. They have won seven All-Ireland Senior Hurling Championships during this period: in 2014, 2015, 2016, 2019, 2020, and two consecutive titles in 2021. 

However, it's important to note that sports results can change rapidly with each new season. Therefore, for the most current information, you may want to check the latest records or news from a reliable source.
```

Granted Kilkenny are the greatest team of all time but not in the last 10 years. Also, there are hallucinations in the response as theier last title is in 2015 and not in 2016, 2019 and 2020.
I put this down to the data the model was trained on and also some hallucination involved. It can't be perfect at everything, right?! However, this doesn't help my pride about my county. As I don't have the resources or time to retrain the model, I was wondering how I could set the record straight.
I then remmembered that Granite 32. provides RAG capability. This got me think that I could supplement documentation RAG style inaddition to my prompt. Lets try this out in the nect section.

### Prompt model with RAG

This time when we ask the question `What team is the most successful hurling team in the last 10 years?`, lets add additional context from WikipediA and [Limerick county hurling team](https://en.wikipedia.org/wiki/Limerick_county_hurling_team) page.

I am going to use the following text from the page:

```
The 2018 season concluded with Limerick winning the 2018 All-Ireland SHC, the team's first since 1973, with a 3–16 to 2–18 point defeat of Galway in the final.The team built on this success, winning the NHL in 2019, 2020 and 2023,
the Munster SHC in 2019, 2020, 2021 and 2022 and the All-Ireland SHC again in 2020, 2021 and 2022.
Munster Senior Hurling Championship 2023, All Ireland Hurling Championship 2023 to be forever remembered the team to join the Cork hurling Champions of the 40s and the Kilkenny hurling Champions of the 2000s to complete 4 in a row.
```

The code snippet now looks as follows:

```py
import openai

api_key = "ollama"
base_url = "http://localhost:11434/v1" # Ollama server hosting model
default_model = "granite3.2:8b"

openai_client = openai.OpenAI(
    base_url=base_url, api_key=api_key
)

# prompt = "What team is the most successful hurling team in the last 10 years?"

prompt = """<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: March 14, 2025.
You are Granite, developed by IBM. You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.

In your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list.<|end_of_text|>
<|start_of_role|>documents<|end_of_role|>Document 0
The 2018 season concluded with Limerick winning the 2018 All-Ireland SHC, the team's first since 1973, with a 3–16 to 2–18 point defeat of Galway in the final.The team built on this success, winning the NHL in 2019, 2020 and 2023, the Munster SHC in 2019, 2020, 2021 and 2022 and the All-Ireland SHC again in 2020, 2021 and 2022. Munster Senior Hurling Championship 2023, All Ireland Hurling Championship 2023 to be forever remembered the team to join the Cork hurling Champions of the 40s and the Kilkenny hurling Champions of the 2000s to complete 4 in a row.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>What team is the most successful hurling team in the last 10 years?<|end_of_text|>
<|start_of_role|>assistant {"citations": true}<|end_of_role|>"""

result = openai_client.completions.create(
    model=default_model,
    prompt=prompt,
    temperature=0.0,
)
results = []
print("\n\n")
for choice in result.choices:
    print(f"{choice.text}\n")
```

Now the response from the model is far more appealing to me as the answer is accurate because of being able to compliment the prompt with extra source:

```shell
Based on the information provided, Limerick has been one of the most successful hurling teams in Ireland over the past decade. They won the All-Ireland Senior Hurling Championship (AHSC) in 2018 and then went on to win it again in 2020, 2021, and 2022 <co>1</co>. Additionally, they secured the National Hurling League (NHL) titles in 2019, 2020, 2023, and the Munster Senior Hurling Championship (MSHC) in 2019, 2020, 2021, and 2022. Their achievement of winning four consecutive AHSC titles from 2020 to 2023 places them among the elite hurling teams in history, alongside Cork's dominance in the 1940s and Kilkenny's in the 2000s.

# Citations:
<co>1</co> Document 0: "The 2018 season concluded with Limerick winning the 2018 All-Ireland SHC, the team's first since 1973, with a 3–16 to 2–18 point defeat of Galway in the final.The team built on this success, winning the NHL in 2019, 2020 and 2023, the Munster SHC in 2019, 2020, 2021 and 2022 and the All-Ireland SHC again in 2020, 2021 and 2022. Munster Senior Hurling Championship 2023, All Ireland Hurling Championship 2023 to be forever remembered the team to join the Cork hurling Champions of the 40s and the Kilkenny hurling Champions of the 2000s to complete 4 in a row."```

Now,  we are given "Thinking" of the model behind the answer in-addition to the answer or response. Exactly what we asked for. 

However, in the code snippet above you can see that we had to make a substantial and intricate change to the prompt for the model to use RAG. This is not something we want to have to do every time.  Wouldn't it be great if we could be abstracted from this and only require to provide the question or prompt. This is where the [granite-io library](https://github.com/ibm-granite/granite-io) comes into play. Follow onto the next section to see how it can simplify your task.

### Use granite-io to help prompting the model

Lets now look at how the code snippet looks when using `granite-io` library to do the heavy lifting:

```
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

model_name = "granite3.2:8b"
io_processor = make_io_processor(
    model_name, backend=make_backend("openai", {"model_name": model_name})
)
question = "Find the fastest way for a seller to visit all the cities in their region"
messages = [UserMessage(content=question)]

# With Thinking
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(messages=messages, thinking=True)
)
print("------ WITH THINKING ------")
print(">> Thoughts:")
print(outputs.results[0].next_message.reasoning_content)
print(">> Response:")
print(outputs.results[0].next_message.content)
```

The output returned from the model can now be broken up, thanks to how the library returns it:

```shell
------ WITH THINKING ------

>> Thoughts:
This problem seems to be a variant of the well-known Traveling Salesman Problem (TSP), which involves finding the shortest possible route that visits each city once and returns to the origin. However, the question asks for the "fastest" way, which suggests we might need to consider additional factors like road conditions, traffic patterns, real-time updates, etc., than just geographical distance. 

To solve this effectively, I'd need to incorporate elements of optimization algorithms (like Genetic Algorithms or Ant Colony Optimization typically used for TSP), real-time data feed integration (for traffic information), and perhaps machine learning models to predict optimal routes based on historical data.

Given the complexity, creating a fully functional system would require developing software and integrating various APIs, which is beyond the scope of this platform. Instead, I'll outline a high-level approach and suggest existing tools or services that can be utilized.

>> Response:
To find the fastest way for a seller to visit all cities in their region, you'd essentially want to solve a variant of the Traveling Salesman Problem (TSP), considering factors beyond just geographical distance such as real-time traffic conditions. Here’s a high-level approach to tackle this:

1. **Data Collection**: Gather data for all cities in the region including their exact locations (latitude and longitude) and any initial known information about road infrastructure. 

2. **Real-Time Traffic Integration**: Incorporate real-time traffic data. This can be achieved by integrating APIs from services like Google Maps API, HERE Routing API, or OpenStreetMap with Overpass Turbo, which all offer robust traffic and route optimization features. These tools not only provide distance and time for different routes but also account for current traffic conditions.

3. **Historical Traffic Data**: Utilize historical data to predict future traffic patterns. Machine learning models, such as regression or neural network models, can be trained on past traffic data to forecast optimal travel times at different times of the day/week.

4. **Route Optimization Algorithm**: Implement an optimization algorithm. Genetic Algorithms, Ant Colony Optimization, or even simpler algorithms like nearest neighbor combined with 2-opt improvement can be used here. These will help in finding the most efficient route considering traffic conditions and possibly other constraints (like avoiding toll roads, certain routes due to construction etc.).

5. **Iterative Refinement**: Start with a preliminary route, perhaps generated by a heuristic method (like nearest neighbor), then iteratively refine it using the optimization algorithm, recalculating based on updated real-time traffic information.

6. **Consider Vehicle Specifics**: If possible, factor in specific vehicle details like average speed, fuel efficiency (to minimize stops) if aiming to minimize travel time/cost instead of just distance.

7. **User Interface**: Finally, create a user-friendly interface where the seller can input destinations and receive optimized routes. This could be a mobile app or a web application, providing options like daily re-planning based on updated traffic forecasts.

While it's not feasible to develop such a system in this text-based environment, the outlined strategy provides a roadmap for software developers looking to create a solution for efficient city-to-city travel planning, integrating real-time data and optimization techniques.
```

## What value does the `granite-io` library give you?

Well, if we use the example of questioning the model about the `Find the fastest way for a seller to visit all the cities in their region`, and also asking how it arrived at its answer:

- Input Processor: Require setting parameter `thinking=True` and the library will generate the prompt required for the model to understand that it needs to also provide Thinking/Reasoning. Remember the ugly prompt! Input Processor)
- Output Processor: It parses the model output and separates the thinking and response parts for you
- IO Processor: It wraps the input and output capability in 1 processor for you. **Note:** the library will also provide the capability to use input processors and output processor independent of the IO processor.


 
