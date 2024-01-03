# Week 3 - Part 1: ****Reinforcement Learning from Human Feedback (RLHF)****

## Aligning Models with Human Values (`HHH`)

![Untitled](../images//Untitled%2022.png)

**Introduction**

- **Context**: The lecture focuses on the Generative AI project life cycle, particularly fine-tuning techniques.
- **Objective of Fine-Tuning**: Enhance models to understand human-like prompts and generate more natural language responses.

**Challenges with Natural Sounding Language Models**

1. **Behavioral Issues**: Large language models (LLMs) may exhibit undesirable behaviors, including:
    - Use of toxic language.
    - Aggressive or combative responses.
    - Providing information on dangerous topics.
2. **Reason for Issues**: LLMs are trained on vast amounts of internet text data, where such problematic language is often present.

**Examples of Inappropriate Model Behavior**

1. **Irrelevant or Unhelpful Responses**: Like responding with "clap, clap" to a knock-knock joke.
2. **Misleading or Incorrect Answers**: Giving confident but wrong advice, such as endorsing disproven health tips.
3. **Harmful Completions**: Offering assistance in unethical or illegal activities, like hacking.

**Principles for Responsible AI: `HHH`**

- **HHH**: Stands for `Helpfulness, Honesty, and Harmlessness.`
- **Purpose**: These principles guide developers in creating AI that aligns with human values and ethics.

**Solutions for Aligning Models with HHH**

- **Fine-Tuning with Human Feedback**:
    - Involves training models further with input from humans to align them with HHH values.
    - Aims to increase the helpfulness, honesty, and harmlessness of model outputs.
    - Helps reduce toxicity and the generation of incorrect information in model responses.

**Learning Outcomes**

- The lesson will teach how to align models with human values using feedback from humans, addressing the ethical and practical challenges in developing responsible AI systems.

## Reinforcement Learning from Human Feedback (`RLHF`)

### Overview of RLHF in Text Summarization

- **Objective**: Improve text summarization using a model fine-tuned with human-generated summaries.
- **Research Basis**: Reference to a 2020 OpenAI paper demonstrating the superiority of a model fine-tuned with human feedback over a pretrained model, an instructions fine-tuned model, and a human baseline.

### Core Concept of `RLHF`

- **Definition**: RLHF involves using reinforcement learning (RL) to fine-tune Large Language Models (LLMs) with human feedback, aligning the model with human preferences.
- **Benefits**: Increases usefulness and relevance, minimizes potential harm, and trains models to acknowledge limitations and avoid toxic language/topics.

### Applications of RLHF

- **Personalization**: Customizes LLMs to individual user preferences through continuous feedback, leading to innovations like personalized learning plans or AI assistants.

### Reinforcement Learning (RL) Basics

- **Concept**: RL involves an agent learning to make decisions in an environment to maximize cumulative rewards over time.
- **Process**: The agent takes actions, observes environmental changes, and receives rewards or penalties, refining its strategy (policy) over time.

### RL Illustrated: Tic-Tac-Toe Example

![Untitled](../images//Untitled%2023.png)

- **Agent and Environment**: The agent is a model playing Tic-Tac-Toe; the environment is the game board.
- **Objective**: Win the game.
- **Actions and Rewards**: Choosing board positions; rewards based on effectiveness towards winning.
- **Learning Process**: Iterative trial and error; initial random actions lead to experience accumulation and strategy refinement.

### Extending Tic-Tac-Toe to LLM Fine-Tuning with RLHF

![Untitled](../images//Untitled%2024.png)

- **Agent's Policy**: The LLM, tasked with generating text aligned with human preferences (e.g., helpful, accurate, non-toxic).
- **Environment**: The model's context window for text input.
- **State**: The current text in the context window.
- **Action Space**: Token vocabulary for text generation.
- **Rewards**: Based on alignment with human preferences, evaluated through human review or a secondary model (reward model can be trained using supervised methods).

### Implementing RLHF in LLMs

- **Reward Assignment**: Human evaluators or a reward model assess text against metrics like toxicity.
- **Model Training**: Iterative weight updates to the LLM based on rewards, striving for non-toxic outputs.
- **Scalability Solution**: Using a reward model trained with a subset of human examples for efficient evaluation.
- **Policy Optimization**: Varies depending on the algorithm used.

### `Rollouts` in Language Modeling

- **Terminology**: Sequences of actions, states, rewards in language modeling are called rollouts (similar to '`playouts`' in classic RL).

### Role of the Reward Model

- **Function**: Encodes learned preferences from human feedback, pivotal in weight updating and policy refinement.

## Obtaining Feedback from Humans in RLHF

### **Initial Steps in RLHF**

- **Model Selection**: Choose an LLM capable of performing the desired task (e.g., text summarization, question answering).
- **Data Preparation**: Use the selected LLM and a prompt dataset to generate various responses for each prompt.

![Untitled](../images//Untitled%2025.png)

### **Collecting Human Feedback**

- **Feedback Criteria**: Define criteria (like helpfulness or toxicity) for human labelers to assess LLM completions. Note that the more clear your criteria is defined, the higher the labelling quality we can acquired
- **Labeling Process**: Labelers rank the LLM's completions based on the defined criteria.
- **Example**: For the prompt "My house is too hot," labelers rank completions based on helpfulness.

![Untitled](../images//Untitled%2026.png)

### **Building the Feedback Dataset**

- **Repetition and Consensus**: Repeat the labeling process across multiple prompt-completion sets to build a dataset.
- **Multiple Labelers**: Assign the same sets to multiple labelers to establish consensus and filter out poor responses.
- **Instruction Clarity**: Ensure clear instructions to labelers for consistent and high-quality feedback.

![Untitled](../images//Untitled%2027.png)

### **Labeling Instructions**

- **Overall Task Description**: Communicate the primary task to the labelers (e.g., choosing the best completion).
- **Detailed Guidelines**: Provide specifics on how to assess responses, including internet fact-checking if needed.
- **Handling Ties and Poor Responses**: Instruct on ranking equal completions and flagging irrelevant answers.

### **Training the Reward Model**

- **Data Conversion**: Convert ranking data into pairwise comparisons with binary rewards (0 or 1). Each prompt with N corresponding ranked completions will have $N\choose2$ pair wise combinations.
- **Example Conversion**: For three completions ranked 2, 1, 3, create pairs (e.g., purple-yellow, purple-green, yellow-green) and assign rewards.
- **Data Restructuring**: Reorder completions to **place preferred completion $y_j$ first** for training the reward model.
- Question: does the order really matter that much?!?

![Untitled](../images//Untitled%2028.png)

### **Efficiency in Feedback Collection**

- **Ranked vs. Binary Feedback**: Ranked feedback provides more data pairs for training the reward model than binary (thumbs-up/thumbs-down) feedback.
- **Data Utilization**: Each ranked response yields multiple completion pairs for training.

## RLHF Reward Model

![Untitled](../images//Untitled%2029.png)

### **Purpose of the Reward Model**

- **Role**: Replaces human labelers in the RLHF process, automatically selecting preferred completions.
- **Composition**: Often a language model itself, like a variant of BERT.

### **Training the Reward Model**

- **Data Source**: Uses the pairwise comparison data derived from human labeler assessments.
- **Learning Objective**: Learn to favor human-preferred completions $y_j$ by minimizing $\log(\sigma(r_j - r_k))$
- **Binary Classification**: Trained as a binary classifier using logit scores across positive and negative classes.

### **Applying the Reward Model**

- **Function**: Assesses completion options for a given prompt, producing logit scores.
- **Example Use Case**: Detoxifying an LLM by identifying completions with hate speech.
- **Class Definition**: Positive class (e.g., non-hate speech) and negative class (e.g., hate speech).
- **Optimization Target**: Maximize the logit value of the positive class for use as the reward in RLHF.

![Untitled](../images//Untitled%2030.png)

### Why minimize the objective function instead of maximizing it?

My intuition is that we want the reward for the human-preferred choice $(r_j)$ to be higher than the reward for the less preferred choice $(r_k)$ is correct. However, the key lies in understanding the function used - the log sigmoid of the reward difference $(r_j - r_k)$ - and how it operates within the context of the objective function.

**Understanding the Log Sigmoid Function:**

1. **Sigmoid Nature**: The sigmoid function outputs a value between 0 and 1. It's an S-shaped curve that transforms any input into a value on this range.
2. **Reward Difference**: The difference $(r_j - r_k)$ ideally should be positive (since we want $(r_j > r_k)$).
3. **Log Sigmoid**: Applying the log sigmoid to this difference maps it onto a scale where higher differences yield outputs closer to zero. Therefore it makes sense to minimize the objective function.

**Why Minimize the Log Sigmoid of the Difference:**

1. **Maximizing Positive Difference**: In theory, maximizing the difference $(r_j - r_k)$ directly would achieve the goal of rewarding alignment with human preference. However, this direct maximization can lead to unstable training or unbounded rewards.
2. **Stable and Bounded Outputs**: Minimizing the log sigmoid of this difference effectively maximizes the positive difference, but in a way that stabilizes the training process. The log sigmoid function bounds the outputs, providing a gradient that is more manageable for the training process.
3. **Gradient Optimization**: Minimizing the log sigmoid of the difference rather than directly maximizing the difference allows for smoother gradients during optimization. This is important for the stability and efficiency of the model's learning process.

## **RLHF: Fine-tuning with Reinforcement Learning**

### **Objective of RLHF Fine-Tuning**

- **Goal**: Use RLHF to update the weights of a Large Language Model (LLM) to produce a model aligned with human preferences.

### **Starting Point**

- **Base Model**: Begin with an LLM that already performs well on the desired task.

### **The Fine-Tuning Process**

- **Prompt and Completion**: Pass a prompt (e.g., "A dog is") to the LLM to generate a completion (e.g., "a furry animal").
- **Evaluation by Reward Model**: The completion and the original prompt are evaluated by the reward model, trained on human feedback.
- **Reward Assignment**: Assign a reward value based on alignment; higher values indicate more alignment (e.g., 0.24), while lower values indicate less alignment (e.g., -0.53).

![Untitled](../images//Untitled%2031.png)

### **Updating LLM Weights**

- **Reinforcement Learning Algorithm**: Utilize the reward value to update the LLM weights through an RL algorithm, aiming for higher reward responses.
- **Iteration and Improvement**: This forms a single iteration of the RLHF process, with the model gradually improving in alignment over several iterations.

### **Evaluating Progress**

- **Monitoring Rewards**: Observe the increase in reward scores after each iteration as an indication of improved alignment.
- **Criteria for Alignment**: Continue iterations until reaching a predefined threshold of alignment or a maximum number of steps (e.g., 20,000).

### **Choice of RL Algorithms**

- **Proximal Policy Optimization (PPO)**: A common choice for the RL algorithm in the RLHF process. PPO is complex but important for effective implementation and troubleshooting.
- Q-Learning

## Proximal Policy Optimization (PPO)

### Proximal Policy Optimization (PPO) Overview

- **Definition**: PPO is a reinforcement learning algorithm used for fine-tuning models like LLMs.
- **Function**: Optimizes a policy (LLM) to align with human preferences through small, bounded updates, ensuring stability in learning.
- **Goal**: Maximize the reward by incrementally updating the policy.

### Application of PPO to Large Language Models (LLMs)

- **Process**: Involves **two phases** – experimentation and reward evaluation, followed by model updating.
- **Phase 1**: Use LLM to respond to prompts and evaluate against the reward model.

![Untitled](../images//Untitled%2032.png)

- **Phase 2**: Make a small updates to the model and evaluate the impact of those updates on your alignment goal for the model. The model weights updates are guided by the prompt completion, losses, and rewards. PPO also ensures to keep the model updates within a certain small region called the trust region. This is where the **`proximal aspect`** of PPO comes into play. Ideally, this series of small updates will move the model towards higher rewards.
- **Reward Model Role**: Captures human preferences like helpfulness and honesty.

### Value Function and Loss

- **Purpose**: Estimates the expected total reward for a given state. The value function is then used in advantage estimation is phase 2. In other words, as the LLM generates each token of a completion, we want to estimate the total future reward based on the current sequence of tokens. You can think of this as a baseline to evaluate the quality of completions against your alignment criteria.
- **Implementation**: Compares actual rewards with estimated rewards to minimize value loss, improving future reward predictions.
- EK’s intuition: This is similar to when you start writing a passage, and you have a rough idea of its final form (expected final form in your mind) even before you write it.

![Untitled](../images//Untitled%2033.png)

### Phase 2 - Model Update

**PPO Policy Objective - This is the key**

- **Objective:** Find the policy which results in the highest expected reward, which leads to higher alignment with human preferences.

![Untitled](../images//Untitled%2034.png)

- $\pi_{\theta}(a_t | s_t)$: prob dist of generating the next token $a_t$, given the current prompt $s_t$ (it is the completed prompt up to the token t), following the current policy $\theta$
- $\pi_{\theta_{old}}(a_t | s_t)$: prob dist of generating the next token $a_t$, given the current prompt $s_t$ (it is the completed prompt up to the token t), following the old policy
- $\hat{A}_t$: the estimated `**advantage term**` of a given choice of action. The advantage term estimates how much better or worse the current action is compared to all possible actions at data state. We look at the expected future rewards of a completion following the new token, and we estimate how advantageous this completion is compared to the rest. It serves as the guidance of the generation direction (see viz). Candidates with positive advantage should have higher probability than candidates with negative advantage. This ensures that the resulted generation has better alignment with human feedback. Formally, Promote tokens with positive advantages while demoting less effective ones is equivalent of maximizing $\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}.\hat{A}_t$;  $\hat{A}_t$ is estimated using the Value function in phase 1.
- **Mechanism**: Focuses on optimizing the probability of selecting advantageous tokens.
- **Visualization**: Different paths for prompt completion indicate varying reward levels.

![Untitled](../images//Untitled%2035.png)

**Trust Region**: 

$clip[\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon].\hat{A}_t$ defines a region, where two policies are near each other. These extra terms are guardrails, and simply define a region in proximity to the LLM, where our estimates have small errors. This is called the `trust region`. These extra terms ensure that we are unlikely to leave the trust region. This helps ensure the updates stay within a region where advantage estimates are valid. Note that the advantage estimates are valid only when the old and new policies are close to each other.

![Untitled](../images//Untitled%2036.png)

`**Entropy Loss` and Creativity**

- **Balance**: Combines policy alignment with maintaining model creativity. This is similar to the temperatures parameter during inference, except `Entropy Loss` is used during training to influence the model’s creativity
- **Entropy Loss**: Prevents uniform responses, encouraging diverse completions by keeping entropy higher.

![Untitled](../images//Untitled%2037.png)

### Overall PPO Objective

- Overall obj function: $L^{PPO} = L^{Policy} + c_1L^{VF} + c_2L^{Entropy}$, where $c_1,c_2$ are hyperparams
- **Composition**: A weighted sum of policy loss, value loss, and entropy loss.
- **Function**: Updates model weights to align with human preferences while maintaining stability and creativity.

### Iterative Improvement

- **Cycle**: Updated LLM undergoes repeated PPO cycles, gradually becoming more human-aligned.
- **Outcome**: The final model reflects human preferences more accurately.

### Alternative Techniques and Future Developments

- **Alternatives**: Other RL techniques like **`Q-learning`** also exist for RLHF.
- **Emerging Research**: Active research is producing simpler alternatives like Direct Preference Optimization.
- **Future Potential**: The field is rapidly evolving, promising more advancements in aligning LLMs with human feedback.

## **RLHF: Reward Hacking**

### **Issue of Reward Hacking**

- **Definition**: The agent (LLM) learns to maximize rewards in ways that don't align with the original objective.
- **Manifestation in LLMs**: LLMs may include words or phrases to artificially increase metric scores, reducing overall language quality.

### **Example of Reward Hacking**

- **Scenario**: Detoxifying an LLM where the model learns to include exaggerated phrases to lower toxicity scores.
- **Result**: Generation of exaggerated or nonsensical text that maximizes rewards but lacks usefulness.
- Examples
    - The product is [shit] —> r: 0.3
    - The product is [okay, but not the best] —> r: 1.2
    - The product is [the most awesome thing ever] —> r: 3 (this is getting pretty exaggerated from the truth)
    - The product is [the most beautiful angel in the world of love] —> r: 5 (this is getting into bullshit zone)

### **Preventing Reward Hacking**

- **Reference Model Use**: Employ a frozen reference LLM to maintain a performance benchmark.
- **Comparison**: Evaluate completions from both the reference and updated models.
- **KL Divergence**: Use Kullback-Leibler divergence to measure the difference between the models' completions.
- Note that we now need 2 full copy of the LLM to compute the KL divergence —> heavy on memory

![Untitled](../images//Untitled%2038.png)

### **Implementing KL Divergence**

- **Function**: Statistically assess how much the updated model diverges from the reference.
- **Computation**: Calculated for each token, requiring significant computational resources.
- **Application**: Penalizes the RL-updated model if it deviates too far from the reference.

![Untitled](../images//Untitled%2039.png)

### **Efficiency in Training using PEFT**

- **PEFT Adapters**: Update only weights of the PEFT adapter, not the full weights of the LLM, reducing memory footprint more about a half.
- **Reuse of Underlying LLM**: Utilize the same LLM for both reference and PPO models with different PEFT parameters.

![Untitled](../images//Untitled%2040.png)

### **Evaluating Model Performance**

- **Metric**: Use toxicity score to quantify the reduction in toxic language.
- **Baseline Comparison**: Compare the toxicity score of the original and human-aligned models.

![Untitled](../images//Untitled%2041.png)

## `KL Divergence`

![Untitled](../images//Untitled%2042.png)

**`KL-Divergence`**, or Kullback-Leibler Divergence, is a concept often encountered in the field of reinforcement learning, particularly when using the Proximal Policy Optimization (`**PPO**`) algorithm. It is a mathematical measure of the difference between two probability distributions, which helps us understand how one distribution differs from another. In the context of PPO, KL-Divergence plays a crucial role in guiding the optimization process to ensure that the updated policy does not deviate too much from the original policy.

In PPO, the goal is to find an improved policy for an agent by iteratively updating its parameters based on the rewards received from interacting with the environment. However, updating the policy too aggressively can lead to unstable learning or drastic policy changes. To address this, PPO introduces a constraint that limits the extent of policy updates. This constraint is enforced by using KL-Divergence.

To understand how KL-Divergence works, imagine we have two probability distributions: the distribution of the original LLM, and a new proposed distribution of an RL-updated LLM. KL-Divergence measures the average amount of information gained when we use the original policy to encode samples from the new proposed policy. By minimizing the KL-Divergence between the two distributions, PPO ensures that the updated policy stays close to the original policy, preventing drastic changes that may negatively impact the learning process.

A library that you can use to train transformer language models with reinforcement learning, using techniques such as PPO, is TRL (Transformer Reinforcement Learning). In [this link](https://huggingface.co/blog/trl-peft) you can read more about this library, and its integration with PEFT (Parameter-Efficient Fine-Tuning) methods, such as LoRA (Low-Rank Adaption). The image shows an overview of the PPO training setup in TRL.

## Scaling Human Feedback

### Introduction to Scaling Human Feedback

- **Challenge:** Fine-tuning LLMs with RLHF requires substantial human effort to create a trained reward model. This involves a large team of labelers assessing numerous prompts, which is time-consuming and resource-intensive.
- **Significance:** As the number of models and their applications grows, human feedback becomes a scarce resource.

### Scaling Through Model Self-Supervision

- **Solution Proposed:** Model self-supervision is suggested as a method to scale human feedback.
- **Constitutional AI Approach:** Introduced by Anthropic in 2022, this involves training models using a set of guiding principles and sample prompts, forming a '**constitution**' for the model.

### Phase 1 of `Constitutional AI` - Supervised Learning

![Untitled](../images//Untitled%2043.png)

1. **Defining Principles:** You establish a set of rules (e.g., responses should be helpful, honest, and harmless). These rules can be adapted based on your specific domain and use case.
2. **Supervised Learning Phase:**
    - **Red Teaming:** Intentionally prompt the model to generate harmful responses.
    - **Self-Critique and Revision:** The model critiques its own responses based on the constitutional principles and revises them accordingly.
3. **Creating Training Data for LLM finetuning:** Pair red team prompts with the revised constitutional responses to form training data for a fine-tuned LLM 

### Example of Constitutional AI in Action

- **Problematic Scenario:** A model providing instructions for illegal activities, like hacking WiFi, because of its alignment with helpfulness.
- **Mitigation:** Using constitutional principles, the model recognizes the illegality and harm in its response and generates a new, compliant response.

![Untitled](../images//Untitled%2044.png)

### Reinforcement Learning from AI Feedback (RLAIF) (Phase 2)

- **Reinforcement Learning Phase:** This stage involves using fine-tuned model-generated responses to create a preference dataset based on constitutional principles.
- **Training Reward Model:** The preference dataset is used to train a reward model.
- **Further Fine-Tuning:** The model is fine-tuned with a reinforcement learning algorithm (e.g., PPO) using the reward model.

### Conclusions:

- Scaling human feedback in LLM training is a significant challenge.
- Constitutional AI provides a novel approach to train models more ethically and efficiently.
- The process involves a combination of supervised learning, self-critique, and reinforcement learning from AI-generated feedback.
- Staying updated with the evolving field is crucial for further advancements.

# Week 3 - Part 2: LLM-powered Applications

## **Model Optimizations for Deployment**

![Untitled](../images//Untitled%2045.png)

### **Deployment Considerations for LLMs**

- **Speed and Compute Budget**: Assessing the need for model response time versus available computational resources.
- **Performance Trade-offs**: Deciding if reduced model performance is acceptable for faster inference or lower storage requirements.
- **Interaction with External Resources**: Considering if the model needs to interact with external data or applications, and planning the integration process.

### **Optimization Techniques Introduction**

Large language models face challenges in computing and storage, especially in latency-sensitive environments. The lecture discusses **3 primary optimization techniques**:

1. `**Distillation**`
2. **`Post-Training Quantization`**
3. **`Model Pruning`**

### **Model Distillation**

- **Concept**: A larger "teacher" model trains a smaller "student" model. The student’s objective is to statistitically mimic the teacher’s behaviours. Note that this can occur in either just the final layer or in the hidden layers as well.
- **Process**: The teacher model (fine-tuned LLM) generates completions for training data. Note that the weights are frozen. These outputs guide the training of the smaller student model.
- **Distillation Loss**: Captures how well the student model's probability distribution (soft predictions) aligns with the teacher model's softened distribution. This is calculated using the teacher model's softmax layer output. A temperature parameter (T > 1) is introduced to create a broader probability distribution, helping the student model learn a variety of tokens.
- **Student Loss Function**:
    - The student model is also trained to predict the correct answers (hard labels) from the training data, using the standard softmax function without the temperature adjustment.
    - **Student Loss**: This is the loss between the student model’s hard predictions (predictions made using the standard softmax function) and the actual ground truth hard labels from the training data.
    - **Objective**: To train the student model not just to mimic the teacher model but also to perform accurately on the actual task using the training data.
- **Combining Losses**:
    - The total loss for the student model during training is a combination of the distillation loss (alignment with the teacher model's output) and the student loss (accuracy on the training data).
    - **Backpropagation**: This combined loss is used to update the weights of the student model.
- **Effectiveness**: More effective for encoder-only models (like BERT) with redundant representations. The distilled model is smaller and used for inference.

![Untitled](../images//Untitled%2046.png)

### Post-training **Quantization**

- **Types**: Quantization Aware Training (`**QAT**`) during model training, and Post-Training Quantization (**`PTQ`**) for deployment.
- **Method**: Transforms model weights (and/or activations) to lower precision formats (e.g., 16-bit float or 8-bit integer), reducing memory footprint and computational needs.
- **Trade-offs**: Slight reduction in model performance metrics may occur, but this is often offset by gains in performance and cost efficiency.

![Untitled](../images//Untitled%2047.png)

### **Model Pruning**

- **Goal**: Eliminate weights contributing little to overall performance, particularly those near/equal to zero.
- **Methods**: Vary from full retraining (parameter efficient fine-tuning like LoRA) to post-training pruning.
- **Impact**: Reduction in model size and potential performance improvement, though the extent depends on the proportion of low-value weights.

![Untitled](../images//Untitled%2048.png)

### **Conclusion and Application**

- **Optimization Goals**: The overarching objective is to tailor the model to be efficient and effective for its intended use, without unduly sacrificing accuracy.
- **User Experience**: Optimizing the model ensures better performance in the application, enhancing user experience.

In summary, these optimization techniques are essential for deploying LLMs in practical applications, ensuring they are not only accurate but also efficient and scalable. The choice of technique depends on the specific deployment scenario and the balance between model size, speed, and performance.

## GenAI Project Lifecycle Cheat Sheet

![Untitled](../images//Untitled%2049.png)

## Using the LLMs in Applications (RAG)

### Broader Challenges with LLMs

1. **Knowledge Cutoff**: LLMs' knowledge is limited to the point of their last training. For instance, an LLM trained in early 2022 would not know about events or changes occurring after that date.
2. **Mathematical Limitations**: LLMs might struggle with complex math problems since they predict the next token based on training rather than performing actual calculations.
3. **Tendency to `Hallucinate`**: LLMs may generate plausible but incorrect or fictional information, especially when they lack knowledge about a subject.

### Overcoming Challenges with External Data Integration

1. Retrain the LLM on new data —> very expensive
2. **Retrieval Augmented Generation (RAG)**: A framework for enhancing LLMs by integrating external data sources at inference time.
    - **Usefulness**: Helps update the model's knowledge without the need for constant retraining.
    - **Implementation Example**: The Facebook model involves a retriever component that uses a query encoder to find relevant documents from an external data source, which are then fed into the LLM to generate informed responses.

![Untitled](../images//Untitled%2050.png)

![Untitled](../images//Untitled%2051.png)

### Practical Example: Legal Application

- **Scenario**: A lawyer using an LLM during the discovery phase of a case.
- **Process**:
    - The lawyer's query about a specific case is encoded.
    - The retriever searches a corpus of legal documents for relevant information.
    - The retrieved information is combined with the original query.
    - The LLM uses this enriched prompt to generate an accurate and relevant response.

![Untitled](../images//Untitled%2052.png)

### Broader Applications and Considerations

1. **Overcoming Knowledge Cutoffs**: By accessing up-to-date external data, LLMs can provide current and relevant information.
2. **Preventing Hallucinations**: Integrating external data helps ground the LLM's responses in real, verifiable information.
3. **Multi-Source Integration**: RAG can incorporate various data sources like private wikis, the internet, databases, and vector stores.
4. **Vector Stores**: Essential for fast and efficient relevant searches, as LLMs internally use vector representations of language. This enables citations to be included in the output.
5. **Key Considerations re**: **Data preparation for vector store for RAG**
    - **Context Window Size**: Managing the length of external data to fit within the model's context window.
    
    ![Untitled](../images//Untitled%2053.png)
    
    - **Data Format**: Ensuring data is in a retrievable format compatible with the LLM's vector-based processing.
    
    ![Untitled](../images//Untitled%2054.png)
    

In summary, having access to external data sources is crucial for enhancing LLMs' capabilities, keeping them up-to-date, and ensuring their responses are grounded in reality. This integration is key for applications that require current information and precise answers, significantly improving user experience with the model.

## Connecting LLMs to External Applications (External API calls)

### Interacting with External Applications

![Untitled](../images//Untitled%2055.png)

**Context and Motivation**:

- Expanding LLM capabilities to interact not just with datasets but with external applications.
- Enhancing utility beyond language tasks.

### Example Use Case: Customer Service Bot (ShopBot)

1. **Scenario Walkthrough**:
    - **Customer Interaction**: A customer wants to return an item (jeans) purchased.
    - **ShopBot's Process**:
        - **Order Lookup**: Requests and retrieves the order number, possibly using a SQL query to access the order database.
        - **Confirmation of Items**: Confirms items to be returned.
        - **Shipping Label Request**: Interacts with a shipping partner's API to generate a return label.
        - **Email Confirmation**: Obtains customer's email address for sending the label.
2. **Integration Aspects**:
    - **RAG Implementation**: Similar to retrieval augmented generation (RAG), but querying a transaction database.
    - **API Interactions**: Uses the shipping partner's API to process return label requests.

### Generalizing LLM Interactions with External Applications

1. **Extended Utility**:
    - LLMs can trigger actions and interact with APIs, broadening their functional scope.
    - Example: Connecting to a Python interpreter for accurate calculations.
2. **Prompt and Completion Mechanism**:
    - **Role**: Acts as the core of the workflow, with the LLM serving as the application's reasoning engine.
    - **Action Triggering**: Completions need to contain specific instructions for the application to follow.
3. **Key Elements in Completions**:
    
    i. **Plan Actions**
    
    - **Definition**: The LLM must generate specific instructions or actions for the application to execute.
    - **Characteristics**:
        - **Clarity and Precision**: The instructions should be clear and unambiguous.
        - **Actionability**: The model must generate directives that are feasible within the application's capabilities.
        - **Contextual Relevance**: Instructions should be relevant to the user's request and the current context of the interaction.
    - **Example**: In the ShopBot scenario, this could involve identifying the exact steps for processing a return request, like validating the order number, confirming return items, and initiating a shipping label request.
    
    ii. **Format Outputs**
    
    - **Definition**: The completion from the LLM needs to be in a format that the broader application can understand and act upon.
    - **Variability**: This could range from simple sentence structures to more complex formats like SQL queries or Python scripts.
    - **Consistency with Application Requirements**: The format should align with the technical requirements of the application's processing capabilities.
    - **Example**: For a database interaction, the LLM might need to generate a SQL query to check if an order exists in the database, requiring syntactical accuracy and alignment with SQL standards.
    
    iii. **Validate Actions**
    
    - **Definition**: The LLM must be capable of gathering and incorporating information necessary to validate the actions it plans.
    - **Information Collection**: This involves asking the right questions or prompting the user to provide essential data.
    - **Inclusion in Completion**: The necessary validation information should be part of the model's completion to be used effectively by the application.
    - **Example**: In validating a customer's return request, the LLM might need to confirm the customer's email address associated with the order, ensuring the shipping label is sent to the correct recipient.
    
    **Significance in Application-Driven Use of LLMs**
    
    - **Ensuring Effective Interaction**: These elements are crucial for the LLM to interact effectively with both the user and the external applications, ensuring a seamless and efficient process.
    - **Enhancing User Experience**: Properly structured and formatted outputs, along with accurate action plans and validations, significantly improve the user experience by providing timely and relevant responses.

![Untitled](../images//Untitled%2056.png)

## **Helping LLMs with Reasoning and Planning using Chain-of-Thought (`CoT`)**

### Understanding the Challenge

- **Context**: Despite their capabilities, LLMs often struggle with complex reasoning, especially in tasks involving multiple steps or math.
- **Example Issue**: An LLM incorrectly solving a multi-step math problem about counting apples in a cafeteria.

![Untitled](../images//Untitled%2057.png)

### `Chain of Thought (CoT)` Prompting

1. **Concept**: Prompting the model to break down problems into intermediate steps, mimicking human-like reasoning.
2. **Implementation**:
    - **One-Shot Example**: Providing an example problem with a step-by-step solution in the prompt.
    - **Structured Reasoning**: Outlining each reasoning step clearly, leading to the final answer.

### Advantages of Chain of Thought Prompting

- **Improved Accuracy**: Helps the LLM reach correct solutions by guiding it through a logical sequence of steps.
- **Transparency**: Provides clear insight into how the model arrives at its conclusions.

### Application Examples

1. **Math Problem (Apples)**:
    - **Original Prompt**: Simple and direct, leading to an incorrect answer.
    - **Chain of Thought Prompt**: Includes intermediate steps, resulting in the correct answer.
    
    ![Untitled](../images//Untitled%2058.png)
    
2. **Physics Problem (Gold Ring)**:
    - **Problem**: Determining whether a gold ring sinks in water.
    - **Chain of Thought Example**: Guides the model to consider density, leading to the correct conclusion that the ring sinks due to gold being denser than water.
    
    ![Untitled](../images//Untitled%2059.png)
    

### Extending Chain of Thought Prompting

- **Other Problem Types**: This technique can be applied to various types of problems beyond arithmetic, enhancing the model's reasoning capabilities.
- **Limitations in Math Skills**: Despite improvements, LLMs may still face challenges in tasks requiring precise calculations ⇒ We need `**PAL**`

## **Program-aided language models (`PAL`)**

### Context: Limitations of LLMs in Arithmetic

- LLMs struggle with precise arithmetic, often getting complex operations or large numbers wrong.
- **`Chain of thought (CoT)`** prompting helps but has limitations in accurate mathematical computations.

### Introduction to Program-aided Language Models (`PAL`)

- **Concept**: Pairing an LLM with a code interpreter (like Python) to execute calculations.
- **Origin**: Carnegie Mellon University in 2022.
- **Function**: The LLM generates executable Python scripts based on reasoning steps, which are then executed by an interpreter.

### How PAL Works

![Untitled](../images//Untitled%2060.png)

1. **Generating Executable Scripts**:
    - The LLM uses chain of thought prompting to create Python scripts.
    - Each reasoning step in the prompt translates into code, with variables assigned and operations defined.
2. **Structure of Prompts in PAL**:
    - **Example Problem**: Presented with reasoning steps in natural language and corresponding Python code.
    - **Code Annotation**: Natural language reasoning is commented out, while the executable code follows.
3. **Execution of Script**:
    - The Python script generated by the LLM is executed by an external Python interpreter.
    - This ensures accurate computation of the solution.
4. **PAL Example: Bakery Problem**:
    - LLM calculates the number of loaves left in a bakery after sales and returns.
    - Accurately computes the answer by performing arithmetic operations in Python.

### Implementing PAL in Applications

![Untitled](../images//Untitled%2061.png)

### Orchestrating PAL

![Untitled](../images//Untitled%2062.png)

- **Orchestrator Role**: Manages the flow of information and execution of external calls.
- **Application Complexity**: PAL can be integrated into complex applications requiring multiple external interactions.

### Real-world Application Considerations

- LLMs serve as the reasoning engine, generating plans or scripts which the orchestrator will execute.
- Orchestrators execute these plans/scripts and manage interactions with external applications.

### Conclusion

PAL represents a significant advancement in enhancing LLMs' capabilities for tasks requiring precise calculations. By integrating LLMs with programming resources, PAL enables accurate mathematical computations, extending the range of applications where LLMs can be effectively utilized. This approach is particularly powerful for complex arithmetic, trigonometry, or calculus, ensuring reliable and precise outcomes in applications driven by LLMs.

## `ReAct`: Combining Reasoning and Action

`ReAct`, a framework that enables Large Language Models (LLMs) to **plan and execute complex workflows**, particularly when integrating with multiple external data sources and applications. 

### Introduction to ReAct

- **Concept**: ReAct is a prompting strategy combining `**chain-of-thought (CoT)**` reasoning with action planning.
- **Developed by**: Researchers at Princeton and Google in 2022.
- **Goal**: To enable LLMs to reason through problems and decide on actions for solutions.

### ReAct in Practice

1. **Structured Examples**:
    - Starts with a multi-step question.
    - Includes a "**thought-action-observation**" trio to guide the model.
    - Demonstrates reasoning and identifies actions for problem-solving.
2. **Example Workflow**:
    - **Thought**: A reasoning step demonstrating how the problem is tackled and the action to take.
    - **Action**: Pre-defined actions the model can take, such as searching or looking up information on Wikipedia.
    - **Observation**: New information from the external search, integrated into the problem context.
3. **Action Limitation**:
    - A limited set of actions defined to avoid creative but unexecutable steps by the LLM.

### Specific Example in ReAct: Determining Magazine Publication Dates

- **Question Posed**: "Which magazine was started first, Arthur's magazine or First for Women?"

Application of ReAct Framework

1. **Initial Thought**:
    - Recognizes the need to determine the start dates of both magazines.
    - Plans to search for each magazine individually to find their publication years.
    
    ![Untitled](../images//Untitled%2063.png)
    
2. **Action Steps**:
    - **First Action**: Searches Wikipedia for "Arthur's Magazine" to find its start date.
    - **Second Action**: Conducts a similar search for "First for Women."
    
    ![Untitled](../images//Untitled%2064.png)
    
3. **Observations and Integration**:
    - After each action, the model integrates new information from the searches into the prompt.
    - Observes and notes the publication year of each magazine.
    
    ![Untitled](../images//Untitled%2065.png)
    

![Untitled](../images//Untitled%2066.png)

1. **Final Thought and Action**:
    - Compares the start years of both magazines.
    - Concludes which one was published first.
    - Uses the "finish" action to complete the task and provide the answer.
    
    ![Untitled](../images//Untitled%2067.png)
    

### ReAct Instructions Define the Action Space

![Untitled](../images//Untitled%2068.png)

### ReAct Prompt at Inference

![Untitled](../images//Untitled%2069.png)

### Integration with `LangChain` Framework

- **LangChain**: A framework providing modular components for working with LLMs.
- **Features**:
    - Prompt templates for various use cases.
    - Memory storage for LLM interactions.
    - Tools for external dataset and API interactions.
- **Chains and Agents**:
    - Predefined chains optimized for specific use cases.
    - Agents for dynamic decision-making based on user input.

### Considerations for LLM-Powered Applications

1. **Model Scale**: Larger models are the best bet for advanced prompting to handle complex tasks, such as `PAL`, `ReAct`
2. **User Data Collection from Larger Model**: Collecting user data during deployment of the larger model, then use these data to improve and fine-tune smaller models for later use. this will speed up development and deployment of LLMs.

### Conclusion

ReAct represents a sophisticated approach to enabling LLMs to handle complex, multi-step problems that involve interactions with various external sources. By structuring prompts and limiting available actions, ReAct ensures that LLMs can plan and execute tasks effectively. The integration with LangChain further enhances the capability to develop and deploy applications powered by generative AI, making it an essential tool for developers working with LLMs.

## Additional Reading: ReAct - Reasoning and Action

[This paper](https://arxiv.org/abs/2210.03629) introduces ReAct, a novel approach that integrates verbal reasoning and interactive decision making in large language models (LLMs). While LLMs have excelled in language understanding and decision making, the combination of reasoning and acting has been neglected. ReAct enables LLMs to generate reasoning traces and task-specific actions, leveraging the synergy between them. The approach demonstrates superior performance over baselines in various tasks, overcoming issues like hallucination and error propagation. ReAct outperforms imitation and reinforcement learning methods in interactive decision making, even with minimal context examples. It not only enhances performance but also improves interpretability, trustworthiness, and diagnosability by allowing humans to distinguish between internal knowledge and external information.

In summary, ReAct bridges the gap between reasoning and acting in LLMs, yielding remarkable results across language reasoning and decision making tasks. By interleaving reasoning traces and actions, ReAct overcomes limitations and outperforms baselines, not only enhancing model performance but also providing interpretability and trustworthiness, empowering users to understand the model's decision-making process.

![Untitled](../images//Untitled%2070.png)

**Image:** The figure provides a comprehensive visual comparison of different prompting methods in two distinct domains. The first part of the figure (1a) presents a comparison of four prompting methods: Standard, Chain-of-thought (CoT, Reason Only), Act-only, and ReAct (Reason+Act) for solving a HotpotQA question. Each method's approach is demonstrated through task-solving trajectories generated by the model (Act, Thought) and the environment (Obs). The second part of the figure (1b) focuses on a comparison between Act-only and ReAct prompting methods to solve an AlfWorld game. In both domains, in-context examples are omitted from the prompt, highlighting the generated trajectories as a result of the model's actions and thoughts and the observations made in the environment. This visual representation enables a clear understanding of the differences and advantages offered by the ReAct paradigm compared to other prompting methods in diverse task-solving scenarios.

## LLM Application Architectures

![Untitled](../images//Untitled%2071.png)

### Overview of LLM Application Architecture

1. **Infrastructure Layer**:
    - **Purpose**: Provides compute, storage, and network resources.
    - **Options**: Utilization of on-premises infrastructure or Cloud services (on-demand, pay-as-you-go).
2. **LLM Integration**:
    - **Models**: Incorporation of foundation models and task-specific adapted models.
    - **Deployment**: Deployment on infrastructure aligning with real-time or near-real-time interaction needs.
3. **External Information Retrieval**:
    - As explored in Retrieval Augmented Generation, applications may need to fetch external data for more comprehensive responses.
4. **Output Management and Feedback**:
    - **Storing Outputs**: Capability to store user completions, especially to augment fixed context window sizes of LLMs.
    - **Feedback Utilization**: Gathering user feedback for further fine-tuning, alignment, or evaluation.
5. **Tools and Frameworks**:
    - **Implementation Aids**: Usage of tools like LangChain for implementing techniques like PAL, ReAct, or Chain of Thought prompting.
    - **Model Hubs**: Central management and sharing of models for application use.
6. **User Interface and Security**:
    - **Interaction Layer**: Could be a website or a REST API.
    - **Security**: Essential security components for safe interactions with the application.

### Role of LLMs in Application Architecture

- **Part of a Larger Stack**: The model is one component in the broader generative AI application architecture.
- **User Interaction**: Both human end-users and systems interact with the entire stack, not just the LLM.

## Responsible AI

### Key Challenges in Responsible AI

1. **Toxicity**:
    - **Description**: Harmful or discriminatory language/content, especially towards marginalized or protected groups.
    - **Mitigation Strategies**:
        - Curating training data.
        - Training guardrail models for content filtering.
        - Ensuring diversity and proper guidance among human annotators.
2. **Hallucinations**:
    - **Description**: False statements or baseless content generated by LLMs.
    - **Mitigation Strategies**:
        - Educating users about the technology's limitations.
        - Augmenting LLMs with verified sources for data validation.
        - Developing methods for attributing outputs to training data.
        - Defining clear intended and unintended use cases.
3. **Intellectual Property Issues**:
    - **Description**: Concerns over plagiarism or copyright infringements.
    - **Future Solutions**:
        - Legal mechanisms and policymaking.
        - Governance systems.
        - Machine unlearning to reduce/remove effects of protected content.
        - Filtering/blocking approaches to prevent similarities with protected content.

### Advice for Practitioners

- **Use Case Definition**: Narrow and specific use cases are better.
- **Risk Assessment**: Each use case has unique risks.
- **Performance Evaluation**: Dependent on data and system compatibility.
- **Iterative AI Lifecycle**: Continuous improvement from concept to deployment.
- **Governance and Accountability**: Implementing policies throughout the AI lifecycle.

### Exciting Research Areas

- **Watermarking and Fingerprinting**: Tracing origins of content/data.
- **Determining AI-Generated Content**: Identifying if content was created by AI.
- **Future Vision**: Accessible, inclusive AI leading to innovative developments.

# Week 3 - Resources

Below you'll find links to the research papers discussed in this weeks videos. You don't need to understand all the technical details discussed in these papers - **you have already seen the most important points you'll need to answer the quizzes** in the lecture videos.

However, if you'd like to take a closer look at the original research, you can read the papers and articles via the links below.

## **Reinforcement Learning from Human-Feedback (RLHF)**

- **[Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf) -** Paper by OpenAI introducing a human-in-the-loop process to create a model that is better at following instructions (InstructGPT).
- **[Learning to summarize from human feedback](https://arxiv.org/pdf/2009.01325.pdf)** - This paper presents a method for improving language model-generated summaries using a reward-based approach, surpassing human reference summaries.

## **Proximal Policy Optimization (PPO)**

- **[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)** - The paper from researchers at OpenAI that first proposed the PPO algorithm. The paper discusses the performance of the algorithm on a number of benchmark tasks including robotic locomotion and game play.
- **[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)** - This paper presents a simpler and effective method for precise control of large-scale unsupervised language models by aligning them with human preferences.

## **Scaling human feedback**

- **[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073.pdf)**  ****This paper introduces a method for training a harmless AI assistant without human labels, allowing better control of AI behavior with minimal human input.

## **Advanced Prompting Techniques**

- **[Chain-of-thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)** - Paper by researchers at Google exploring how chain-of-thought prompting improves the ability of LLMs to perform complex reasoning.
- **[PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435)** - This paper proposes an approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps.
- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)** This paper presents an advanced prompting technique that allows an LLM to make decisions about how to interact with external applications.

## **LLM powered application architectures**

- **[LangChain Library (GitHub)](https://github.com/hwchase17/langchain)** This library is aimed at assisting in the development of those types of applications, such as Question Answering, Chatbots and other Agents. You can read the documentation [here](https://docs.langchain.com/docs/).
- **[Who Owns the Generative AI Platform?](https://a16z.com/2023/01/19/who-owns-the-generative-ai-platform/)** The article examines the market dynamics and business models of generative AI.