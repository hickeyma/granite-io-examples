# Granite 3.2 Reasoning Example

Granite 3.2 provided new capability like Thinking or reasoning. <TODO: add to>

## Prompt the model

Lets start with sample code snippet below to ask `Granite 3.2 8b` model the following question `Find the fastest way for a seller to visit all the cities in their region`:

```py
import openai

api_key = "ollama"
base_url = "http://localhost:11434/v1" # Ollama server hosting model
default_model = "granite3.2:8b"

openai_client = openai.OpenAI(
    base_url=base_url, api_key=api_key
)

prompt = "Find the fastest way for a seller to visit all the cities in their region"

result = openai_client.completions.create(
    model=default_model,
    prompt=prompt,
)
results = []
print("\n\n")
for choice in result.choices:
    print(f"{choice.text}\n")
```

The model will generate output similar to the following:

```shell
To find the fastest route for a seller to visit all cities in their region, you're essentially looking for the solution to the Traveling Salesman Problem (TSP). TSP is an NP-hard problem in combinatorial optimization, meaning there's no known efficient algorithm that can solve it exactly for large inputs. However, there are several approaches and heuristics that can provide near-optimal solutions:

1. **Exact Algorithms**: These methods guarantee finding the optimal solution but become impractical for a large number of cities due to their high computational complexity. Examples include branch and bound, dynamic programming, and integer linear programming formulations.

2. **Heuristic Methods**: These are faster than exact algorithms but do not guarantee an optimal solution. They're often used in practice because they can handle larger problem sizes. Some popular heuristics for TSP include:

   - **Nearest Neighbor Algorithm**: Start from a city and at each step, move to the nearest unvisited city until all cities have been visited. This is simple but not always efficient.
   
   - **2-Opt Algorithm**: Improve an initial tour by repeatedly swapping two edges if it results in a shorter tour.
   
   - **3-Opt Algorithm**: Similar to 2-Opt, but considers three edges for possible swap.
   
   - **Genetic Algorithms**: These are evolutionary algorithms inspired by the process of natural selection. They work with a population of candidate solutions and iteratively improve them through processes like mutation, crossover, and selection.
   
   - **Simulated Annealing**: This is a probabilistic technique for approximating the global optimum of a given function. It's particularly useful when dealing with large, complex problems where local optima are common.

3. **Approximation Algorithms**: These provide solutions that are guaranteed to be within a certain factor of the optimal solution. The Christofides algorithm is an example, providing a tour no longer than 1.5 times the length of the minimum spanning tree plus the edge weights connecting the remaining vertices.

4. **Specialized Software and APIs**: There are software tools and online services that can solve TSP instances efficiently using advanced algorithms or by leveraging cloud computing power. Examples include Concorde TSP Solver, Google OR-Tools, and various web-based TSP solvers.

To implement these methods effectively:

1. **Data Preparation**: Gather accurate distance data between all city pairs. This can be done using geographical coordinates and a suitable distance calculation method (e.g., Haversine formula for spherical Earth).

2. **Choose an Approach**: Depending on the number of cities, computational resources, and required solution quality, select an appropriate method or combination of methods.

3. **Implementation**: Implement the chosen approach using a suitable programming language (e.g., Python, Java) or leverage existing software/APIs.

4. **Optimization**: Fine-tune parameters and potentially combine multiple approaches to improve solution quality and efficiency.

5. **Validation**: Verify the results by comparing them with other methods or known optimal solutions for smaller instances.
```

This is great but I'd like to better understand the model's thinking or reasoning behind he answer it provided. It might be just me but I'm a curious type!
Follow onto the next example on how to ask the model about their thinking.

# Prompt model and ask for context as well

This time when we ask the question `Find the fastest way for a seller to visit all the cities in their region`, lets also ask the model how it arrived at its answer:

```py
import openai

api_key = "ollama"
base_url = "http://localhost:11434/v1" # Ollama server hosting model
default_model = "granite3.2:8b"

openai_client = openai.OpenAI(
    base_url=base_url, api_key=api_key
)

# prompt = "Find the fastest way for a seller to visit all the cities in their region"

prompt = """<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: March 13, 2025.
You are Granite, developed by IBM. You are a helpful AI assistant.
Respond to every user query in a comprehensive and detailed way. You can write down your thoughts and reasoning process before responding. In the thought process, engage in a comprehensive cycle of analysis, summarization, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. In the response section, based on various attempts, explorations, and reflections from the thoughts section, systematically present the final solution that you deem correct. The response should summarize the thought process. Write your thoughts after 'Here is my thought process:' and write your response after 'Here is my response:' for each user query.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Find the fastest way for a seller to visit all the cities in their region<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>"""

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

Now the response from the model is far more detailed and nuanced around how it arrived at its answer:

```shell
Here is my thought process:

The problem described seems to be a variant of the well-known Traveling Salesman Problem (TSP), which is an NP-hard optimization problem. The goal is to find the shortest possible route that visits each city once and returns to the origin city. However, finding the absolute optimal solution for a large number of cities is computationally intensive and impractical due to its exponential time complexity. 

Given this, we need to consider heuristic or approximation algorithms that can provide a "good enough" solution in a reasonable amount of time. Common methods include:

1. Nearest Neighbor Algorithm: Start from a city and at each step, move to the nearest unvisited city. This is simple but not guaranteed to find the best route.

2. Genetic Algorithms or Simulated Annealing: These are more sophisticated metaheuristic methods that can provide near-optimal solutions for larger datasets but require careful tuning of parameters and computational resources.

3. Christofides Algorithm: This is a specific algorithm designed to find a route within twice the optimal, making it efficient for certain types of problems.

Considering the need for speed, we might lean towards simpler methods like Nearest Neighbor or more complex but potentially faster methods like Genetic Algorithms, depending on the number of cities and computational resources available. 

Here is my response:

To find the fastest way for a seller to visit all cities in their region (a problem similar to the Traveling Salesman Problem - TSP), we need to balance optimality with computational efficiency. Given the constraint of speed, here's a suggested approach:

1. **Nearest Neighbor Algorithm**: This is a straightforward heuristic method that can provide a quick solution. Here’s how it works:

   - Start from any city (let's call this City A).
   - From City A, go to the nearest unvisited city (City B).
   - From City B, go to the nearest unvisited city (could be City C, D, etc.).
   - Continue this process until all cities are visited.
   - The route ends by returning to the starting city (City A).

This method is fast and easy to implement but doesn't guarantee the shortest possible route. It's a good choice when speed is crucial and approximate solutions are acceptable.

2. **Genetic Algorithm**: For potentially better results, you could use a more sophisticated approach like Genetic Algorithms:

   - **Initialization**: Randomly select a set of initial routes (each representing a 'chromosome' in genetic terms).
   - **Fitness Function**: Define a fitness function that evaluates how 'good' each route is (e.g., total distance). The goal is to minimize this value.
   - **Selection, Crossover, Mutation**: These are the core operations of Genetic Algorithms:
     - Selection: Choose routes for reproduction based on their fitness scores. Better routes have a higher chance of being selected.
     - Crossover: Combine parts of two routes to create new ones.
     - Mutation: Randomly alter some routes to maintain diversity and prevent premature convergence.
   - **Termination**: Repeat the selection, crossover, and mutation steps over several generations until you reach a satisfactory solution or a predefined computational limit.

Genetic Algorithms can provide better solutions than Nearest Neighbor but require more computational resources and careful parameter tuning.

3. **Christofides Algorithm**: This is a specific algorithm designed to find a route within twice the optimal for certain types of problems (Euclidean TSP in metric spaces). It's efficient and provides near-optimal results, but its implementation can be complex.

**Considerations**:

- The number of cities: For a very small number (<20), Nearest Neighbor might suffice. For larger datasets, more sophisticated methods like Genetic Algorithms or Christofides (if applicable) would be preferable.
  
- Computational resources: Simpler methods require less computational power but may provide less optimal results. More complex algorithms can offer better solutions at the cost of increased processing time and memory usage.

- Time constraints: If time is of the essence, a quick approximation (like Nearest Neighbor) might be preferred over a more computationally intensive method that guarantees better, though not necessarily the best, solution. 

In conclusion, the 'fastest' method depends on the specific context and requirements of the problem at hand. For a swift yet approximate solution, the Nearest Neighbor Algorithm is recommended. For situations where slightly better results are acceptable and more computational resources are available, Genetic Algorithms would be a suitable choice.
```

 
