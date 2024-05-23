# How to do retrieval augmented generation (RAG) right!

- date: 2024-05-23

- source: https://aisc.substack.com/p/how-to-do-retrieval-augmented-generation?triedRedirect=true

### Summary of Lecture Notes on Retrieval Augmented Generation (RAG)

#### **Introduction**
- **Purpose of the Article**: 
  - Discuss the widespread adoption of Retrieval Augmented Generation (RAG) for large language models (LLMs).
  - Address common client issues with RAG implementations.
  - Highlight the importance of robust software design and tailored solutions.

#### **Understanding the Motivation Behind RAG**
- **Challenges with LLMs**:
  - **Verifiable Facts**: LLMs struggle with accuracy in facts and real-world understanding.
  - **Language Limitation**: They excel at language but not at specialized tasks like mathematics or data analysis.
- **Potential of External Data Integration**:
  - **Personalized Responses**: Combining LLMs with external knowledge sources (databases, APIs, documents) can enhance responses.
  - **Modular Approach**: Separating the linguistic interface from the data layer allows for more precise processing (security, privacy, relevance).

#### **Why RAG?**
- **Limitations of LLMs**:
  - **Knowledge Scope**: LLMs can't cover all factual knowledge due to the complexity of the world and the limitations of their training data.
  - **Factual Inaccuracy**: LLMs can generate outdated or incorrect information due to their reliance on patterns rather than deep understanding.
  - **Inconsistency**: LLMs can produce different results for the same input due to their probabilistic nature, which is problematic for tasks requiring stable outputs.

#### **Benefits of RAG**
- **Enhanced Capabilities**: Integrating LLMs with external data sources can provide more accurate and contextually aware interactions.
- **Granular Control**: Allows for better control over the behavior of the LLM by managing the data layer separately.
- **Reliable Information**: Coupling LLMs with reliable information retrieval systems can address the inherent limitations of LLMs.

#### **Challenges in Implementing RAG**
- **Interfacing Diverse Data Formats**: Ensuring seamless integration of various data sources.
- **Retrieval Quality and Efficiency**: Balancing the quality of retrieved information with the efficiency of the system.
- **Trustworthiness of Data**: Ensuring that external data sources are reliable and up-to-date.

#### **Conclusion**
- **Future Potential**: Properly implemented RAG can unlock advanced, knowledge-intensive interactions across various domains.

#### **Example for Better Understanding**
- **Chatbot Enhancement**:
  - **Without RAG**: A chatbot might provide generic responses based on pre-trained data.
  - **With RAG**: The chatbot can access a company's knowledge base and customer interaction history to give personalized and accurate support.

---

### Key Takeaways
1. **RAG enhances LLMs** by integrating them with external data sources, addressing limitations in knowledge scope, factual accuracy, and consistency.
2. **A modular approach** separates the linguistic interface from the data layer, allowing for precise control and processing.
3. **Implementation challenges** include interfacing diverse data formats, balancing retrieval quality and efficiency, and ensuring data trustworthiness.

By understanding these principles and addressing the challenges, RAG can significantly improve the performance and reliability of language models in real-world applications.

### Summary of Lecture Notes on Validation Driven Development

#### **Introduction**
- **Key Concept**: A one-size-fits-all approach to interfacing LLMs with external data doesn't work.
- **Objective**: Design architecture around specific workflow nuances, mimicking human processes of gathering, analyzing, and communicating information for more reliable and nuanced outputs.

#### **Importance of Validation Driven Development**
- **Basic RAG Architecture**: Good starting point but requires thorough assessment to identify weaknesses.
- **Performance Assessment**: Evaluate across diverse situations to pinpoint issues (e.g., factual queries, consistent themes in creative writing).
- **Targeted Improvements**: Focus on specific shortcomings systematically for efficient enhancement.

#### **Approach to Improvement**
- **Systematic Focus**: Address one significant weakness at a time to avoid wasted effort.
- **Example**: Improve fact-checking capabilities iteratively to build a trustworthy RAG system.

#### **Validation vs. Evaluation vs. Verification**
- **Definitions**:
  - **Evaluation**: Assess overall quality and impact (e.g., system's ability to do math).
  - **Verification**: Ensure the system is built correctly (e.g., correct math answers).
  - **Validation**: Ensure the system meets user needs (e.g., good at math questions users care about).

#### **Evaluation**
- **Challenges**: Traditional metrics like accuracy may not capture nuances of human language interactions.
- **New Metrics**: Include fairness, interpretability, creativity, and impact on downstream tasks (e.g., sales from generated proposals).
- **Iterative Process**: Monitor for performance drift and unintended biases.

#### **Verification**
- **Nuanced Process**: Traditional unit testing may not capture emergent behavior.
- **Techniques**:
  - **Adversarial Testing**: Stress the system with unexpected data.
  - **Fuzzing**: Test for unexpected outputs and biases.
- **Common Issues**:
  - Invalid JSON returns.
  - Schema inconsistencies.
  - Intellectual property leaks.
  - Harmful content generation (e.g., hate speech).
  - Malicious code production.
  - Incorrect function signatures.

#### **Validation**
- **Stochastic Nature**: Difficult to determine if output is reliably good.
- **User Experience**: Important for trust; include process visibility, citations, and warnings.
- **Examples**:
  - Verbose execution process display.
  - Citations for information presented.
  - Communication of verification results.

#### **Recipe for Validation Driven Development**
1. **Understand Evaluation, Validation, and Verification**: Plan how these will be integrated into the development process and gather necessary data.
2. **Implement Guardrails**: Prevent undesirable outcomes.
3. **Metric-Driven Development**: Tailor evaluations to specific use cases.
4. **Formal Verification**: Ensure reliability and safety of LLMs when necessary.

---

### Key Takeaways
1. **Nuanced Architecture**: Design around specific workflow nuances for reliable outputs.
2. **Systematic Improvement**: Focus on targeted, significant weaknesses iteratively.
3. **Validation Driven Development**:
   - **Evaluation**: Assess overall quality and impact with new, relevant metrics.
   - **Verification**: Ensure the system functions as intended using advanced testing techniques.
   - **Validation**: Confirm the system meets user needs with a focus on user experience.
4. **Recipe for Success**: Plan evaluation, validation, and verification from the start, implement guardrails, use metric-driven development, and apply formal verification when necessary.

By adopting a structured approach to validation driven development, you can ensure that your RAG system is reliable, nuanced, and meets the specific needs of its users.

### Summary of Lecture Notes on Modular Retrieval-Augmented Generation (RAG)

#### **Introduction**
- **Key Concept**: The basic architecture provided by vector DB vendors often falls short in achieving "good" performance.
- **Objective**: Define a "good" performance for your use case and measure it with appropriate data.
- **Solution**: Use modular architecture to experiment and find the right combination of components.

#### **Modular RAG Overview**
- **Principle**: Make the architecture as modular as possible to allow flexibility in experimenting with different components.
- **Inspiration**: Design the system to mimic human workflows, breaking down processes into sub-tasks.
- **Iteration**: Expect multiple iterations to refine the architecture, uncovering and addressing edge cases along the way.

#### **Common Modules in Modular RAG**
- **Human Workflow Analogy**: Modules should replicate the steps a human would take in a particular workflow.
- **Flexibility**: Modular design allows for easy modifications and optimizations based on performance assessments.

#### **Common Modules in Various Use Cases**
1. **Data Retrieval Module**:
   - **Function**: Retrieve relevant data from external sources (databases, APIs, documents).
   - **Example**: A chatbot accessing a knowledge base for customer support.

2. **Preprocessing Module**:
   - **Function**: Clean and prepare the retrieved data for the language model.
   - **Example**: Normalizing text data, removing irrelevant information.

3. **Relevance Filtering Module**:
   - **Function**: Filter the retrieved data to ensure it is relevant to the query.
   - **Example**: Using keyword matching or semantic similarity to select relevant documents.

4. **Contextual Integration Module**:
   - **Function**: Integrate the filtered data into the context for the language model.
   - **Example**: Embedding the relevant data into the prompt for the LLM.

5. **Language Model Module**:
   - **Function**: Generate responses based on the integrated context.
   - **Example**: An LLM generating a response to a customer query using the integrated data.

6. **Post-Processing Module**:
   - **Function**: Refine the generated response for clarity, accuracy, and appropriateness.
   - **Example**: Correcting grammar, verifying factual accuracy, ensuring compliance with guidelines.

7. **Evaluation and Feedback Module**:
   - **Function**: Evaluate the performance of the generated responses and provide feedback for improvement.
   - **Example**: Using user feedback and automated metrics to assess response quality.

#### **Example of Modular RAG Implementation**
- **Use Case**: Customer Support Chatbot
  1. **Data Retrieval**: Fetch customer interaction history and product information.
  2. **Preprocessing**: Clean the data, remove irrelevant details.
  3. **Relevance Filtering**: Filter interactions and documents related to the customer’s query.
  4. **Contextual Integration**: Embed the filtered information into the query context.
  5. **Language Model**: Generate a response using the LLM with the embedded context.
  6. **Post-Processing**: Refine the response, correct errors, and ensure appropriateness.
  7. **Evaluation and Feedback**: Evaluate the response quality, gather user feedback, and iterate.

---

### Key Takeaways
1. **Modular Architecture**: Essential for flexibility and experimentation in finding the optimal combination of components.
2. **Human Workflow Analogy**: Design modules to replicate human sub-tasks for more natural and effective system behavior.
3. **Iterative Refinement**: Expect multiple iterations to refine the architecture based on performance assessments and edge cases.
4. **Common Modules**: Data retrieval, preprocessing, relevance filtering, contextual integration, language model, post-processing, and evaluation and feedback.
5. **Example Application**: Modular RAG for a customer support chatbot demonstrates the practical implementation of these principles.

By adopting a modular approach to RAG, you can build a more adaptable, efficient, and effective system tailored to your specific use case.

### Summary of Lecture Notes on Query Expansion and Rewriting, and Information/Example Retrieval

#### **Query Expansion and Rewriting**

**Purpose**:
- Transform the user’s query into a format optimized for searching external knowledge bases or data sources.
- This involves rephrasing the query, identifying keywords, or adding relevant information implicitly stated by the user.

**Techniques**:
1. **Basic Query Expansion**:
   - **Pseudo-Relevance Feedback**: Retrieves documents based on the initial query, identifies keywords from those documents, adds them to the original query, and retrieves them again.

2. **Advanced Techniques for LLMs**:
   - **Hypothetical Document Embeddings (HyDE)**: Creates a hypothetical document relevant to the query, uses its embedding to retrieve nearest neighbor documents, and rephrases the query into better-matched terms.
   - **Step-Back Prompting**: Allows LLMs to perform abstract reasoning and retrieval based on high-level concepts.
   - **Query2Doc**: Creates multiple pseudo-documents using prompts from LLMs and merges them with the original query to form a new expanded query.
   - **ITER-RETGEN**: Combines the outcome of the previous generation with the prior query, retrieves relevant documents, and generates new results, repeating the process multiple times.

**Efficiency Considerations**:
- **Computational Cost**: Large models like GPT-4 can handle query expansion but are computationally expensive.
- **Alternative Methods**:
  - Task-specific fine-tuned small LLMs.
  - Knowledge distillation.
  - Pseudo-relevance feedback (PRF).
  - Small classifiers trained on high-quality data.

**Example**:
- **Query Expansion**: A user asks, "Best practices for secure coding." The system rephrases it to include related keywords like "secure coding guidelines," "code security," and "software security best practices" to improve search results.

#### **Information/Example Retrieval**

**Purpose**:
- Fetch relevant information from external data sources based on the optimized search object created by the query expansion module.

**Techniques**:
1. **Keyword Searching**: Over a set of documents.
2. **Extracting Specific Passages**: From texts or documents.
3. **Querying Structured Databases**: Via APIs or SQL-type queries.
4. **Mixed Approach**: Combining all the above methods.
5. **Calling Expert Models**: Using APIs to access machine learning or statistical models, or even physics/engineering simulations.

**Example Retrieval**:
- **Finding Relevant Examples**: Retrieve similar past examples (e.g., FAQs) to improve the context for the LLM. This can enhance the repeatability and accuracy of responses.

**Example**:
- **Information Retrieval**: For the query "secure coding practices," the system retrieves specific passages from security guidelines, articles on secure coding, and relevant sections from cybersecurity textbooks.
- **Example Retrieval**: Retrieves FAQ examples on secure coding to provide a few-shot learning setup for the LLM, enhancing the quality and consistency of the generated responses.

---

### Key Takeaways
1. **Query Expansion and Rewriting**: Essential for optimizing user queries for better search results. Techniques range from basic (pseudo-relevance feedback) to advanced methods (HyDE, Step-Back prompting, Query2Doc, ITER-RETGEN).
2. **Efficiency Considerations**: Balance between computational cost and performance. Alternatives include smaller, fine-tuned models and methods like PRF.
3. **Information/Example Retrieval**: Fetches relevant information and examples from various sources, enhancing the context for LLM responses.
4. **Practical Application**: Effective query expansion and retrieval techniques improve the accuracy, relevance, and repeatability of LLM outputs.

By implementing these strategies, you can significantly enhance the performance and efficiency of systems relying on LLMs and external data sources.

### Summary of Lecture Notes on Reranking & Post-Processing

#### **Introduction**
- **Goal of Retrieval**: Collect as much potentially relevant information as possible, even if some of it is irrelevant.
- **Challenge**: Initial rankings from different systems may not align with the query's relevance.

#### **Reranking Stage**
- **Purpose**: Determine which chunks or data points are most relevant to the user’s query.
- **Importance**: The language model can only process a limited context to generate accurate responses.

**Techniques for Reranking**:
1. **Simple Semantic Ranking**: Basic method for reranking based on semantic similarity.
2. **Advanced Relevance Scoring**:
   - **Cross-Encoders**: Evaluate relevance by encoding the query and document together.
   - **Query Likelihood Models**: Estimate the likelihood of a query given a document.
   - **Supervised Ranking Models**: Use machine learning to train models on relevance data.

**Process**:
- **Relevance Scores**: Assign higher scores to pertinent information and lower scores to irrelevant data.
- **Prioritization**: Select the most valuable content for the LLM to use in generating responses.

**Consideration**:
- **Context Window Limitation**: Avoid overloading the LLM's context window to prevent the "Lost-in-the-middle" problem.
- **Arrangement Experimentation**: Try different arrangements of relevant information (e.g., top, bottom, alternating) to optimize relevance.

#### **Post-Processing Stage**
- **Objective**: Ensure the retrieved information fits within the LLM’s context limit and is free of irrelevant, redundant, or lengthy texts.

**Methods for Textual Data**:
1. **Summarization**: Condense the retrieved text to its essential points.
2. **Chain-of-Note Generation**: Create a structured summary of the information.

**Methods for Other Data Types**:
- **Specialized Operations**: Perform necessary calculations or transformations to boil down the data to what is strictly needed for the task.

#### **Example Process**

**Reranking**:
1. **Initial Retrieval**: Gather a large set of document chunks related to the query "secure coding practices."
2. **Simple Semantic Ranking**: Rank the documents based on keyword matching and semantic similarity.
3. **Advanced Reranking**: Use a cross-encoder to evaluate each document's relevance to the query.
4. **Score Assignment**: Assign higher scores to documents with detailed and accurate secure coding guidelines.
5. **Arrangement Testing**: Experiment with placing the highest-scored documents at different positions within the context window.

**Post-Processing**:
1. **Summarization**: Condense the top-ranked secure coding documents to highlight key practices.
2. **Chain-of-Note Generation**: Create a structured summary outlining the main points of secure coding.
3. **Specialized Operation**: If applicable, calculate statistics or transform the data into a useful format (e.g., a checklist of practices).

---

### Key Takeaways
1. **Reranking**: Crucial for prioritizing the most relevant information to fit within the LLM's limited context window.
2. **Advanced Techniques**: Employ methods like cross-encoders and supervised ranking models to improve relevance scoring.
3. **Context Window Optimization**: Avoid overloading the context window; experiment with different arrangements of relevant information.
4. **Post-Processing**: Summarize and refine the retrieved information to ensure it is concise and relevant.
5. **Specialized Operations**: Apply specific transformations or calculations to non-textual data to make it suitable for the task.

By implementing effective reranking and post-processing strategies, you can significantly enhance the relevance and efficiency of information provided to LLMs, leading to more accurate and useful responses.

### Summary of Lecture Notes on Prompt Crafting and Generation

#### **Introduction**
- **Objective**: Present the determined content (documents, summaries, notes, processed data, examples) to the LLM in a structured manner to optimize the quality and coherence of the generated responses.

#### **Prompt Crafting**

**Process**:
1. **Content Structuring**:
   - Arrange the retrieved and processed information in a logical and coherent template.
   - Experiment with different data templates and order of information to achieve the desired output.

**Techniques**:
- **Standard Approach**: Generate the output all at once.
- **Advanced Techniques**:
  - **Active Retrieval**: Interleave generation and retrieval in an iterative process.
    - **Steps**:
      1. Generate some output.
      2. Retrieve more context based on the generated text.
      3. Generate more output.
      4. Repeat until a complete response is formed.
    - **Decision Factors**: Number of tokens, completion of textual units, or insufficiency of available context.

**Example**:
- **Scenario**: Crafting a prompt for a secure coding guidelines query.
  - **Template 1**: Start with a brief overview of secure coding practices, followed by specific guidelines retrieved, and end with a summary.
  - **Template 2**: Begin with examples of secure coding, include the main guidelines in the middle, and conclude with additional best practices.

#### **Verification**

**Importance**:
- Ensures the generated response is accurate, coherent, and free from hallucinations or incorrect details.

**Verification Techniques**:
1. **Entity and Number Check**:
   - Verify that entities and numbers mentioned are present in the provided context.
2. **Fact-Checking**:
   - Cross-check generated sentences against the context chunks passed to the LLM.
3. **Linguistic Analysis**:
   - Analyze the output to ensure it meets specific linguistic requirements.

**Response to Verification Outcomes**:
- **Pass**: Proceed with the generated response.
- **Fail**: Different actions based on the use case and severity:
  - Append a warning to the user.
  - Provide feedback to the LLM and ask it to try again.
  - Overlay verified information as citations inline with the generated text.
  - Follow an elaborate policy for verification and system behavior based on the outcome.

**Example**:
- **Verification Scenario**: Secure coding guidelines response.
  - **Entity Check**: Verify that all mentioned coding standards (e.g., OWASP) are present in the context.
  - **Fact-Checking**: Ensure specific recommendations (e.g., input validation techniques) are accurate.
  - **Linguistic Analysis**: Check for clarity and coherence in the response.

---

### Key Takeaways
1. **Prompt Crafting**: 
   - Structure and present content to the LLM effectively.
   - Experiment with different templates and arrangements to optimize output.

2. **Advanced Generation Techniques**:
   - **Active Retrieval**: Iterative process of generation and retrieval for maintaining coherence in long-form text generation.

3. **Verification**:
   - Crucial to ensure accuracy and reliability of the LLM-generated responses.
   - Use various techniques like entity and number checks, fact-checking, and linguistic analysis.
   - Implement appropriate actions based on verification outcomes to maintain response quality.

By following these strategies, you can craft effective prompts and ensure the generated responses from LLMs are accurate, relevant, and coherent.

### Summary of Lecture Notes on Design Patterns

#### **Introduction**
- **Purpose**: Discuss how different modules might come together to build a pipeline for RAG (Retrieval-Augmented Generation).
- **Challenge**: Few well-established patterns exist, and the optimal pattern depends on the specific cognitive process and business workflow.

#### **Design Patterns for RAG Pipelines**

**Module Integration**:
- **Common Order**: Modules typically follow the order discussed in previous sections (query expansion, retrieval, reranking, post-processing, prompt crafting, verification).
- **Workflow Replication**: The pattern should replicate the cognitive process and business workflow, mimicking how humans make decisions based on available information.

**Routing Modules**:
- **Decision-Based Steps**: Human cognitive processes involve decision-making steps based on available information, leading to the creation of routing modules.
- **Function**: Routing modules replicate decision processes with predetermined configurations of modules for each possible decision pathway, giving the system an "agency-like" behavior.

#### **RAG and Agentic Workflows**

**Handling Different Scenarios**:
- **Configuration Variability**: Different configurations of modules are necessary to handle various user query scenarios.
- **Agentic Workflows**:
  - **Definition**: Routing or configuration selecting modules that robustly select the appropriate pipeline for different user queries.
  - **Goal**: To follow validation-driven development, addressing one performance issue at a time.

**Example of Agentic Workflow**:
- **Scenario**: Handling technical support queries.
  - **Routing Module**: Determines whether the query relates to software, hardware, or network issues.
  - **Module Configuration**: Selects the appropriate pipeline (e.g., software-related queries might route to a pipeline emphasizing code snippets and technical documentation).

**Fine-Tuning for Performance**:
- **Challenge**: After adding necessary modules, performance issues may persist.
- **Solution**: Fine-tuning the system to address specific performance problems and improve overall effectiveness.

---

### Key Takeaways
1. **Module Integration**:
   - Follow a logical order based on the discussed modules.
   - Mimic human cognitive processes and business workflows.
   - Utilize routing modules to handle decision-based steps and provide agency-like behavior.

2. **Agentic Workflows**:
   - Employ configuration selecting modules to manage variability in user queries.
   - Focus on validation-driven development to incrementally address performance issues.

3. **Example**:
   - Use routing modules to direct technical support queries to the appropriate pipeline based on the nature of the issue (software, hardware, network).

4. **Fine-Tuning**:
   - Essential for addressing persistent performance issues after initial module integration.
   - Tailor the system to improve accuracy, relevance, and efficiency.

By adopting these design patterns and principles, you can create a flexible, efficient, and effective RAG pipeline tailored to your specific use case and workflow requirements.

### Summary of Lecture Notes on Fine-tuned RAG

#### **Introduction**
- **Modularity Advantage**: Each module in the RAG architecture can be independently fine-tuned or trained to handle specific subtasks effectively.

#### **Fine-Tuning Individual Modules**

**Router Module**:
- **Function**: Acts as a query classifier to select the appropriate pipeline.
- **Example**: A simple model that classifies user queries based on recent interactions, directing them to the correct processing pathway.

**Retrieval and Reranking Modules**:
- **Function**: Handle query-specific retrieval and reranking of information.
- **Example**: Models trained with annotated ranking data to understand the nuances of query relevance and improve accuracy.

**Prompt Crafting Module**:
- **Function**: Craft prompts in a supervised manner for optimal LLM output.
- **Example**: Using RLPrompt, a model trained on pairs of input data and desired output to generate well-structured prompts.

#### **Conclusion**

**Basic Vector Database and LLM Combo**:
- **Limitation**: Inadequate for complex real-world use cases that RAG aims to enhance.

**Complex Workflows**:
- **Requirement**: Workflows demand more than just finding and summarizing text, necessitating additional modules beyond the core RAG architecture.

**Modularity Key**:
- **Flexibility**: Designing the RAG architecture around the specific steps of the workflow being augmented provides immense flexibility.
- **Upfront Investment**: Careful planning can save time, money, and frustration in the long run.

**Validation-Driven Development**:
- **Approach**: Start with a simple RAG system, identify edge cases through rigorous evaluation, and address them incrementally.
- **Principles**: Focus on core principles to build a robust and adaptable RAG system that significantly enhances workflows.

---

### Key Takeaways

1. **Modular Fine-Tuning**:
   - Each module can be independently trained or fine-tuned for its specific task.
   - Examples include query classifiers for routing, models trained with annotated data for retrieval and reranking, and supervised prompt crafting models like RLPrompt.

2. **Limitations of Basic RAG**:
   - A simple vector database and LLM combination is insufficient for complex real-world applications.

3. **Complex Workflow Requirements**:
   - Effective RAG implementations need additional modules to handle the complexity of real-world workflows.

4. **Importance of Modularity**:
   - Provides flexibility and adaptability in the RAG system.
   - Careful design and planning of the RAG architecture around specific workflow steps are crucial.

5. **Validation-Driven Development**:
   - Start simple, evaluate rigorously, and tackle edge cases one by one.
   - Focus on core principles to build a robust, adaptable RAG system that enhances workflows effectively.

**Example**:
- **Scenario**: Enhancing customer support with a fine-tuned RAG system.
  - **Router Module**: Classifies queries into categories (e.g., billing, technical support) based on user interactions.
  - **Retrieval Module**: Trained to fetch relevant documents specific to the query category.
  - **Reranking Module**: Fine-tuned to prioritize the most relevant information.
  - **Prompt Crafting Module**: Uses supervised learning to create effective prompts for generating accurate responses.

By applying these strategies, you can develop a sophisticated, fine-tuned RAG system tailored to meet the specific demands of complex workflows, resulting in significant improvements in performance and usability.

