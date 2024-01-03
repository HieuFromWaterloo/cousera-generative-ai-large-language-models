# Week 2 - Part 1: ****Fine-tuning LLMs with instructions****

## Introduction to Fine-Tuning Large Language Models

### **Overview of Week's Focus**

- **Topics**: Instruction fine-tuning of LLMs and efficient techniques for fine-tuning for specific applications.

### **Instruction Fine-Tuning**

- **Purpose**: Helps LLMs respond appropriately to prompts or instructions.
- **Importance**: While LLMs learn general information from vast text data, they may not inherently know how to follow specific instructions.
- **Breakthrough**: A major advancement in LLMs, enabling them to follow instructions after being trained on general text.
- **Challenge**: Avoiding **catastrophic forgetting**, where the model loses its previously learned information.

### **Techniques to Prevent Catastrophic Forgetting**

- **Approach**: Broaden the range of instruction types during fine-tuning.
- **Goal**: Maintain the LLM's general capabilities while learning specific tasks.

### **Types of Fine-Tuning**

- **`Instruction Fine-Tuning`**: Adapting the model to respond to a variety of instructions.
- **`Application-Specific Fine-Tuning`**: Tailoring the model for specialized applications.

### **Challenges in Traditional Fine-Tuning**

- **Resource Intensive**: Fine-tuning every parameter in a large model requires significant compute and memory resources.

### **Parameter-Efficient Fine-Tuning (`PEFT`)**

- **Objective**: Achieve similar performance with fewer resources.
- **Methods**: Freezing original model weights, adding adaptive layers with smaller memory footprints.
- **Advantages**: Enables training for multiple tasks with cost and memory efficiency.

### **`LoRA` Technique**

- **Concept**: Using low-rank matrices instead of full model weights matrix fine-tuning.
- **Benefits**: Good performance with minimal compute and memory requirements.

### **Practical Considerations in Fine-Tuning**

- **Initial Approach**: Starting with prompting, which may suffice for some applications.
- **PEFT for Enhanced Performance**: Critical for surpassing performance limitations of prompting.
- **Cost vs. Benefit**: Weighing the benefits of using large models against the cost of fine-tuning smaller models.

### **Real-World Applicability**

- **Cost Constraints**: PEFT methods make fine-tuning feasible for users with limited budgets.
- **Data Privacy and Control**: Size-appropriate models are essential for maintaining data within user control.

Fine-tuning LLMs, particularly through instruction fine-tuning and PEFT methods like LoRA, is crucial for enhancing their ability to understand and respond to specific instructions while managing computational resources effectively. This week's content delves into these advanced techniques, highlighting their importance in developing practical, cost-efficient, and application-specific LLMs.

## **Instruction Fine-Tuning Large Language Models (LLMs)**

### **Introduction to Fine-Tuning**

- **Context**: Building on the foundation of transformer networks and the generative AI project lifecycle.
- **Objective**: Learn methods to improve the performance of an existing LLM for specific tasks.

### **Concept of Instruction Fine-Tuning**

- **Purpose**: Enhances an LLM's ability to understand and execute specific instructions.
- **Need**: Addressing the limitations of LLMs in responding to prompts or carrying out zero-shot inference, particularly in smaller models.

### **Limitations of Prompt-Based Strategies**

- **Challenges**: Ineffective for smaller models and consumes valuable context window space.

### **Fine-Tuning vs. Pre-Training**

- **Distinction**: Fine-tuning is a `**supervised learning**` process with labeled examples, contrasting with the **`self-supervised`** pre-training on vast unstructured data.
- **Self-Supervised Learning**: A form of training where the model generates its own labels from the input data, as opposed to relying on external labeling. Unlike **`supervised learning`**, self-supervised learning doesn’t require labeled datasets. Instead, it utilizes large volumes of unlabeled text data – such as books, articles, and websites. For example, it might mask certain words or sentences in a text and then try to predict them based on the surrounding context.
- **Key Techniques in LLMs:**
    - **Masked Language Modeling (`MLM`)**: Used in models like BERT, where random words in a sentence are masked (hidden), and the model predicts these words based on the remaining unmasked words.
    - **Causal (or `Autoregressive`) Language Modeling**: Employed in models like GPT, where the model predicts the next word in a sequence based on the previous words, effectively learning to generate text.

### **`Instruction Fine-Tuning` Methodology**

- **Process**: Training the model with **prompt-completion pairs** that illustrate how to respond to specific instructions.
- **Example Tasks**: Summarization, translation, classification, etc.
- **Full Fine-Tuning**: Involves updating all of the model’s weights, resulting in a new version of the model.

### **Requirements for Fine-Tuning**

- **Resources**: Sufficient memory and compute budget to handle gradients, optimizers, and other components during training.
- **Optimization Techniques**: Utilizing memory optimization and parallel computing strategies.

### **Preparing for Fine-Tuning**

- **Data Preparation**: Converting existing datasets into instruction-based formats using **prompt template libraries**.
- **Data Splitting**: Dividing the dataset into training, validation, and test splits.

### **Fine-Tuning Process**

- **Execution**: Feeding training prompts to the LLM, generating completions, and comparing them with training labels.
- **Loss Calculation**: Using `**cross-entropy**` to calculate loss between token distributions and updating model weights accordingly.
- **Validation and Testing**: Evaluating performance on validation and test datasets to assess improvements.

### **Outcomes of Fine-Tuning**

- **Result**: An instruct model that performs better on specified tasks.
- **Commonality**: Instruction fine-tuning is now the most prevalent method for fine-tuning LLMs.

### Conclusion

Instruction fine-tuning is a crucial method for enhancing the capabilities of LLMs to understand and execute specific tasks more effectively. By using labeled prompt-completion pairs, developers can tailor the LLM to their application needs, overcoming the limitations of base models in understanding and responding to instructions. This process requires careful preparation of training data and consideration of resource constraints, but ultimately leads to more efficient and task-oriented LLMs.

## Fine-Tuning LLMs for a Single Task

### **Context and Purpose**

- **Focus**: Fine-tuning pre-trained large language models (LLMs) for a specific language task.
- **Application**: Tailoring the LLM for tasks like summarization or translation using a task-specific dataset.

### **Effectiveness of Single-Task Fine-Tuning**

- **Data Requirement**: Achieving good results often requires just 500-1,000 examples, contrasting with the billions used in pre-training.
- **Potential Drawback**: Risk of `**catastrophic forgetting**`, where the model loses its capability to perform other tasks effectively.

### **Catastrophic Forgetting in Fine-Tuning**

- **Cause**: Occurs when full fine-tuning modifies the weights of the LLM, leading to improved performance in the fine-tuned task but decreased ability in others.
- **Example**: A model fine-tuned for sentiment analysis may forget how to perform tasks like named entity recognition.

### **Addressing Catastrophic Forgetting**

- **Assessing Impact**: Determine if catastrophic forgetting affects your use case. If the focus is solely on one task, this may not be a concern.
- **Multitask Fine-Tuning**: Fine-tune the model on multiple tasks simultaneously, requiring more examples (50-100,000) and greater compute resources.
- One way to mitigate catastrophic forgetting is by using regularization techniques to limit the amount of change that can be made to the weights of the model during training. This can help to preserve the information learned during earlier training phases and prevent overfitting to the new data.
- PEFT

### **Parameter Efficient Fine-Tuning (`PEFT`)**

- **Approach**: Instead of full fine-tuning, PEFT involves training only a small number of task-specific adapter layers and parameters.
- **Advantage**: Preserves most of the pre-trained weights, thus offering more robustness against catastrophic forgetting.

### **PEFT vs. Full Fine-Tuning**

- **PEFT**: Maintains general capabilities of the LLM while improving performance on a specific task.
- **Full Fine-Tuning**: Risks losing the model's multitasking abilities in favor of excelling at a single task.

### Conclusion

Fine-tuning LLMs for a single task can be highly effective, especially when only a small dataset is available for the specific task. However, this approach may lead to catastrophic forgetting, reducing the model's proficiency in other tasks. Alternatives like multitask fine-tuning and PEFT offer solutions to this problem, with PEFT showing particular promise for maintaining the model's versatility. This focus on fine-tuning methods highlights the balancing act between task-specific optimization and the retention of generalized language abilities in LLMs.

## Multi-Task Instruction Fine-Tuning for Large Language Models (LLMs)

### **Extension of Single Task Fine-Tuning**

- **Concept**: Multi-task fine-tuning involves training a pre-trained model with a dataset comprising examples for multiple tasks.
- **Tasks Included**: Summarization, review rating, code translation, entity recognition, etc.

### **Advantages of Multi-Task Fine-Tuning**

- **Improved Performance**: Enhances model proficiency across various tasks simultaneously.
- **Catastrophic Forgetting**: Helps to avoid the issue where fine-tuning on one task degrades performance on others.

### **Process and Requirements**

- **Training Dataset**: A mixed dataset with prompt-completion pairs for a range of tasks.
- **Data Volume**: Requires a substantial number of examples, often 50-100,000, for effective training.
- **Training Method**: Involves updating model weights based on calculated losses across examples over many epochs.

### **Outcome**

- **Result**: An instruction-tuned model capable of performing well in multiple tasks.

### **Example: FLAN Models**

- **`FLAN` (Fine-tuned Language Net)**: A set of models fine-tuned using various instructions.
- **Variants**: FLAN-T5 and FLAN-PALM, fine-tuned versions of the T5 and PALM models respectively.
- **Training Datasets**: FLAN-T5 trained on 473 datasets across 146 task categories.
- **Utilization**: FLAN models are suitable for general-purpose tasks due to their extensive training.

### **Case Study: FLAN-T5 and Dialogue Summarization**

- **SAMSum Dataset**: Used for training FLAN-T5 in dialogue summarization, consists of messenger-like conversations with summaries.
- **Training Method**: Using prompt templates to turn conversations into instruction-based formats.

### **Custom Fine-Tuning Example**

- **Scenario**: Fine-tuning FLAN-T5 for a customer service chatbot application using a domain-specific dataset (DialogSum).
- **DialogSum Dataset**: Over 13,000 support chat dialogues and summaries, not included in FLAN-T5's training data.
- **Objective**: Improve FLAN-T5's ability to summarize support chat conversations.

### **Fine-Tuning with Custom Data**

- **Approach**: Utilize a company's internal data (e.g., customer support conversations) for more tailored fine-tuning.
- **Advantage**: Helps the model learn company-specific summarization preferences and nuances.

### Conclusion

Multi-task instruction fine-tuning represents a powerful approach to enhance the capabilities of pre-trained LLMs across a variety of tasks, mitigating the risk of catastrophic forgetting inherent in single-task fine-tuning. Models like FLAN demonstrate the effectiveness of this method, achieving proficiency in diverse tasks through extensive and varied training datasets. Custom fine-tuning, as exemplified by the use of the DialogSum dataset, further highlights the potential of tailoring models to specific applications, ensuring more relevant and accurate performance in real-world scenarios.

## Scaling instruct models

[This paper](https://arxiv.org/abs/2210.11416) introduces `FLAN` (Fine-tuned LAnguage Net), an instruction finetuning method, and presents the results of its application. The study demonstrates that by fine-tuning the 540B PaLM model on 1836 tasks while incorporating `**Chain-of-Thought**` Reasoning data, FLAN achieves improvements in generalization, human usability, and zero-shot reasoning over the base model. The paper also provides detailed information on how each these aspects was evaluated.

![Untitled](../images/Untitled%208.png)

Here is the image from the lecture slides that illustrates the fine-tuning tasks and datasets employed in training FLAN. The task selection expands on previous works by incorporating dialogue and program synthesis tasks from Muffin and integrating them with new Chain of Thought Reasoning tasks. It also includes subsets of other task collections, such as T0 and Natural Instructions v2. Some tasks were held-out during training, and they were later used to evaluate the model's performance on unseen tasks.

## Model Evaluation

### **Model Evaluation Context**

- **Challenge**: Assessing the performance of LLMs which produce non-deterministic and language-based outputs.

### **ROUGE Metrics for Summarization**

- **ROUGE-1**:
    - **Focus**: Unigram matches.
    - **Example**: Reference - "It is cold outside"; Generated - "It is very cold outside". High ROUGE-1 scores due to unigram overlap.
    
    ![Untitled](../images/Untitled%209.png)
    
    ![Untitled](../images/Untitled%2010.png)
    
- **ROUGE-2**:
    - **Focus**: Bigram matches.
    - **Example**: Lower scores compared to ROUGE-1 as it considers two-word sequences, revealing less similarity.
    
    ![Untitled](../images/Untitled%2011.png)
    
- **ROUGE-L**:
    - **Focus**: Longest common subsequence.
    - **Example**: Identifies sequences like "It is" and "cold outside" to calculate similarity.
    
    ![Untitled](../images/Untitled%2012.png)
    
- `**ROUGE Hacking` and `ROUGE Clipping`**

![Untitled](../images/Untitled%2013.png)

### **BLEU Score for Translation**

- **Concept**: Averages precision across various n-gram sizes.
- **Example Calculation**:
    - Reference: "I am very happy to say that I am drinking a warm cup of tea."
    - Candidate 1: "I am very happy that I am drinking a cup of tea." - BLEU score of 0.495, indicating moderate similarity to the reference.

### **Limitations and Considerations**

- **ROUGE Limitation**: Can score poorly structured sentences highly if they contain many common words. Additionally, ROUGE does not take order into account.
    - **Example**: "Cold, cold, cold, cold" might score high on ROUGE-1 due to repetition.
- **BLEU Limitation**: Scores can be misleading if the sentence structure varies significantly from the reference despite having similar meaning. A sentence that was actually translated correctly, can still receive a low score, depending on the human reference. [https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)
- **Use in Context**: ROUGE is best for summarization, while BLEU is suited for translation. Neither should be the sole metric for final model evaluation.

### Conclusion

The evaluation of LLMs using metrics like ROUGE and BLEU provides structured ways to assess the similarity of language outputs to human-generated references. However, the nuanced nature of language generation by LLMs necessitates the use of more comprehensive benchmarks for a thorough evaluation. These methods, while useful in specific contexts, have limitations and should be supplemented with advanced evaluation tools for a more accurate and holistic understanding of model performance.

## Benchmarks

### Detailed Summary of LLM Benchmarks for Holistic Evaluation

### **Limitations of Simple Metrics**

- **Context**: Simple metrics like ROUGE and BLEU provide limited insights into LLM capabilities.
- **Need**: More comprehensive evaluation methods are necessary for a holistic understanding of LLM performance.

### **Importance of Selecting Right Evaluation Dataset**

- **Objective**: Accurately assess an LLM’s performance and understand its capabilities.
- **Criteria**: Datasets that isolate specific model skills and focus on potential risks.
- **Consideration**: Ensuring the model hasn't seen the evaluation data during training.

### **Key Benchmarks for LLM Evaluation**

- **GLUE (General Language Understanding Evaluation)**:
    - Introduced in 2018.
    - Comprises tasks like sentiment analysis and question-answering.
    - Aims to promote the development of models that generalize across tasks.
- **SuperGLUE**:
    - Launched as a successor to GLUE in 2019.
    - Includes more challenging tasks and some not covered by GLUE.
    - Focuses on multi-sentence reasoning and reading comprehension.

### **Emergence vs. Benchmarking**

- **Trend**: As models grow larger, their performance on benchmarks like SuperGLUE begins to approach human levels in specific tasks.
- **Observation**: Despite matching human performance in tests, LLMs may not exhibit human-level task performance in general.

### **Recent Advanced Benchmarks**

- **Massive Multitask Language Understanding (MMLU)**:
    - Targets modern LLMs.
    - Requires extensive world knowledge and problem-solving skills.
    - Tests on diverse topics like mathematics, history, and computer science.
- **BIG-bench**:
    - Comprises 204 tasks in various fields.
    - Available in three sizes for cost efficiency.
- **Holistic Evaluation of Language Models (HELM)**:
    - Focuses on transparency and model suitability for specific tasks.
    - Multimetric approach covering fairness, bias, and toxicity.
    - Continuously updated with new scenarios and models.
    - **Metrics Beyond Accuracy**: Includes fairness, bias, toxicity, etc

### Conclusion

Advanced benchmarks like GLUE, SuperGLUE, MMLU, BIG-bench, and HELM provide a more comprehensive framework for evaluating the capabilities of large language models. These benchmarks not only assess basic language understanding but also delve into aspects like fairness, bias, and problem-solving abilities, offering a multidimensional view of model performance. The evolution of these benchmarks and their increasing complexity highlight the ongoing advancements in LLM development and the need for robust, multi-faceted evaluation methods to accurately gauge their capabilities and limitations.

# Week 2 - Part 2: ****Parameter efficient fine-tuning****

## Parameter Efficient Fine-Tuning (`PEFT`)

**1. Introduction to PEFT in LLMs**

- **Context**: Training large language models (LLMs) is resource-intensive.
- **Challenge**: Full fine-tuning of LLMs requires substantial memory for various parameters beyond just the model weights.
- **Memory Allocation**: Necessary for optimizer states, gradients, forward activations, and temporary memory, which can exceed the capacity of consumer hardware.

![Untitled](../images/Untitled%2014.png)

**2. PEFT vs. Full Fine-Tuning**

- **Full Fine-Tuning**: Updates every model weight, leading to high memory requirements.
- **PEFT Approach**: Updates only a small subset of parameters, keeping most LLM weights frozen.
- **Benefits**:
    - Reduces the number of trained parameters (sometimes to just 15-20% of the original LLM).
    - Manages memory requirements more effectively, **often enabling training on a single GPU**.
    - Less prone to `**catastrophic forgetting**`.
    - Solves storage issues related to full fine-tuning for multiple tasks.

**3. Implementation of PEFT**

- **Process**: Trains a small number of weights, significantly reducing the overall footprint.
- **Application**: New parameters are combined with original LLM weights for inference (faster inference as compared to full finetuning), allowing efficient adaptation to multiple tasks.

![Untitled](../images/Untitled%2015.png)

**4. Classes of PEFT Methods**

- **`Selective` Methods**:
    - Fine-tune a subset of original LLM parameters.
    - Involve trade-offs between parameter and compute efficiency.
- **`Reparameterization` Methods**:
    - Modify the original LLM parameters by creating new low rank transformations.
    - Example: `**LoRA**` (Low Rank Adaption).
- **`Additive` Methods**:
    - Keep original LLM weights frozen, introducing new trainable components.
    - Types:
        - **Adapter Methods**: Add trainable layers to model architecture.
        - **Soft Prompt Methods**: Manipulating the input to the LLM rather than altering the model's internal weights or architecture for performance, either by adding trainable parameters to prompt embeddings or retraining embedding weights. Since only a small part of the model (the prompts) is being fine-tuned, it significantly reduces computational and memory requirements.

### Soft Prompt Method vs. Regular Prompt Engineering

- **Soft Prompt Method**:
    - Involves introducing trainable embeddings that are optimized during the training process.
    - These embeddings are not human-readable and are adjusted through the training to become task-specific signals for the model.
    - The method requires additional training but results in prompts that are more finely tuned to specific tasks.
    - **No Separate Network**: The training of these embeddings doesn't require a separate neural network. They are trained as part of the input to the existing LLM.
    - **Efficient Fine-Tuning**: This method enables efficient fine-tuning because it avoids the computational and resource-intensive process of updating the entire LLM. Only the relatively small number of parameters in the embeddings are adjusted.
- **Regular Prompt Engineering**:
    - Relies on carefully crafting text-based prompts that are human-readable (e.g., "Write a summary of the following text:").
    - No training of the prompts is involved; it's more about finding the right wording or structure that works well with the pre-trained model.
    - It's more of a trial-and-error process and doesn't involve altering the model or its inputs in a trainable way.

### Examples of Soft Prompt Method:

**Example 1:** Fine-Tuning for Text Summarization

**Scenario**: You have an LLM and you want to fine-tune it for better performance in text summarization tasks.

**Using Soft Prompt Method**:

1. **Initial Setup**: Start with a set of trainable embeddings that act as a prompt. These are not fixed words but rather a sequence of vectors that the model will learn to associate with the task of summarization.
2. **Training Process**: You train these embeddings by feeding them into the model along with your text summarization training data. The goal is for the model to learn that when it sees these specific embeddings, it should focus on summarizing the content that follows.
3. **Adjusting the Prompt**: During training, the embeddings (soft prompts) are adjusted through backpropagation, just like other model parameters. However, the rest of the LLM remains unchanged.

**Example Prompt**: Let's say your original prompt for summarization is a simple instruction like "Summarize the following text:". In the soft prompt method, this would be replaced with a sequence of trainable embeddings. These embeddings don't have a human-readable form but are vectors that the model learns to associate with the task of summarization.

**Example 2:** Adapting LLM for Legal Document Analysis

**Scenario**: You want your LLM to perform better at analyzing legal documents.

**Using Soft Prompt Method**:

1. **Creating Task-Specific Embeddings**: You introduce a new set of trainable embeddings specifically for legal document analysis.
2. **Training for the Legal Domain**: These embeddings are trained with legal texts, teaching the model to recognize and process legal language and concepts more effectively when these embeddings are present.
3. **Integration in Inference**: When analyzing a legal document, these trained embeddings are used as a prefix or addition to the input, signaling the model to apply its legal analysis capabilities.

## PEFT Method - LoRA (Low Rank Adaption)

**Introduction to LoRA**

- **Definition**: LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique categorized under re-parameterization methods.
- **Purpose**: Reduces the number of parameters that need to be trained during fine-tuning of Large Language Models (LLMs).

**Mechanism of LoRA**

1. **Transformer Architecture**: Involves self-attention and feedforward networks where weights are learned during pre-training.
2. **Freezing Original Weights**: LoRA technique freezes all original model parameters.
3. **Injection of Rank Decomposition Matrices**: Introduces two small matrices alongside the original weights. These matrices are designed such that their product matches the dimensions of the weights they modify.
4. **Training Process**: The smaller matrices are trained using supervised learning, while the original LLM weights remain frozen.
5. **Inference Phase**: For inference, the product of the two low-rank matrices is added to the original frozen weights, effectively updating the model for specific tasks.

![Untitled](../images/Untitled%2016.png)

Can easily be fine tuned to various tasks and storing LoRa weight does not take much of memory. So, it’s very efficient to swap these LoRa in inference time.

![Untitled](../images/Untitled%2017.png)

**Advantages of LoRA**

- **Inference Latency**: Maintains the same number of parameters as the original model, thus not impacting inference latency.
- **Focus on Self-Attention Layers**: Primarily applied to self-attention layers, which contain most of the LLM's parameters, leading to significant savings in trainable parameters.

**Practical Example: Transformer Architecture**

- **Original Transformer Weights**: Dimensions of 512x64, equaling 32,768 trainable parameters.
- **LoRA Implementation**: Using LoRA with rank 8, you train two smaller matrices (8x64 and 512x8), resulting in 4,608 parameters, an 86% reduction.
- **Compute Requirements**: This efficiency allows for training on a single GPU and avoids the need for distributed clusters.

**Performance Comparison Using ROUGE Metric**

- **FLAN-T5 Model**: Focused on dialogue summarization.
- **Baseline ROUGE Scores**: Set for the base FLAN-T5 model.
- **Comparison**: Full fine-tuning shows significant performance improvement over the base model. LoRA fine-tuning also shows substantial improvement, though slightly less than full fine-tuning.
- **Trade-Off**: LoRA trades a small decrease in performance for a significant reduction in the number of trainable parameters and computational resources.

**Choosing the Rank of LoRA Matrices**

- **Research Findings**: A plateau in performance gains for ranks greater than 16.
- **Optimal Range**: Ranks between 4-32 offer a good balance between reducing trainable parameters and maintaining performance.
- **Ongoing Research**: Optimizing the rank choice is an active area of research.

![Untitled](../images/Untitled%2018.png)

**Conclusion**

- **Effectiveness**: LoRA is an effective method for fine-tuning LLMs with reduced computational requirements.
- **Broader Application**: The principles of LoRA are applicable beyond LLMs to models in other domains.

## **PEFT Method - Soft Prompts**

### **Soft Prompts** Intuition

The intuition behind using soft prompts is like giving your robot friend a special, task-specific code that helps it understand and perform better on the task you need help with.

Imagine you have a really smart robot friend who is great at doing all sorts of tasks. But sometimes, you want your robot friend to be even better at a specific task, like helping you with your homework. Now, think of this robot as a big language model (like the ones used in computers to understand and generate language).

Now, let's talk about "`soft prompts`." These are like **special clues or hints or a special instructions** that you give to your robot friend to help it understand exactly what you want it to do. When the model sees these soft prompts, it knows it should perform a specific task, like summarizing text or answering a certain type of question.

**Overview of Soft Prompts**

- **Objective**: Improve model performance without altering the model's weights.
- **Category**: Falls under additive methods in Parameter Efficient Fine Tuning (PEFT).

**Prompt Tuning vs. Prompt Engineering**

- **Prompt Engineering**: Involves manually crafting the language of prompts to improve completions. It's limited by the context window and can require extensive manual effort.
- **Prompt Tuning (Soft Prompts)**: Adds trainable tokens to the prompt, allowing the supervised learning process to optimize their values. This method uses a set of virtual tokens called a soft prompt.

**Mechanism of Prompt Tuning**

1. **Soft Prompt Composition**: Comprises 20 to 100 virtual tokens.
2. **Embedding Integration**: These tokens have the same length as language embedding vectors and are prepended to the input text embeddings.
3. **Training Process**: Unlike full fine-tuning, only the embeddings of the soft prompt are trained while the rest of the LLM's weights are frozen.

![Untitled](../images/Untitled%2019.png)

**Efficiency of Prompt Tuning**

- **Parameter Efficiency**: Trains far fewer parameters compared to full fine-tuning, making it a resource-efficient approach.
- **Flexibility**: Different sets of soft prompts can be trained for various tasks and easily swapped out during inference.
- **Storage**: Soft prompts are small in size, making them storage-efficient.

![Untitled](../images/Untitled%2020.png)

**Performance Comparison**

- **Study**: Research by Brian Lester and collaborators at Google.
- **Findings**:
    - In smaller LLMs, prompt tuning underperforms compared to full fine-tuning.
    - In larger models (around 10 billion parameters), prompt tuning can match the effectiveness of full fine-tuning.
    - Significant improvement over prompt engineering alone.

![Untitled](../images/Untitled%2021.png)

**Interpretability of Learned Virtual Tokens**

- **Characteristics**: The tokens do not correspond to known words but form semantic clusters related to the task.
- **Implication**: Indicates that these prompts learn word-like representations that are task-specific.

**PEFT Methods Covered**

1. **LoRA**: Utilizes rank decomposition matrices for efficient parameter updates.
2. **Prompt Tuning**: Adds trainable tokens to prompts without altering the LLM's weights.

**Recap and Applications**

- **Instruction Fine-Tuning**: Adaptation of foundation models for specific tasks.
- **Evaluation Metrics**: Use of ROUGE and HELM for success measurement.
- **PEFT in Practice**: Minimizes compute and memory resources, reducing fine-tuning costs and accelerating development.

**Advanced Techniques**

- **QLoRA**: Combines LoRA with quantization techniques for further efficiency.
- **Practical Use**: PEFT is widely used for optimizing compute budget and speeding up development processes in natural language use cases.

# Week 2 Resources

## **Multi-task, instruction fine-tuning**

- **[Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)** - Scaling fine-tuning with a focus on task, model size and chain-of-thought data.
- **[Introducing FLAN: More generalizable Language Models with Instruction Fine-Tuning](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html)** - This blog (and article) explores instruction fine-tuning, which aims to make language models better at performing NLP tasks with zero-shot inference.

## **Model Evaluation Metrics**

- **[HELM - Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/)** - HELM is a living benchmark to evaluate Language Models more transparently.
- **[General Language Understanding Evaluation (GLUE) benchmark](https://openreview.net/pdf?id=rJ4km2R5t7)** - This paper introduces GLUE, a benchmark for evaluating models on diverse natural language understanding (NLU) tasks and emphasizing the importance of improved general NLU systems.
- **[SuperGLUE](https://super.gluebenchmark.com/)** - This paper introduces SuperGLUE, a benchmark designed to evaluate the performance of various NLP models on a range of challenging language understanding tasks.
- **[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)** - This paper introduces and evaluates four different measures (ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S) in the ROUGE summarization evaluation package, which assess the quality of summaries by comparing them to ideal human-generated summaries.
- **[Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/pdf/2009.03300.pdf)** - This paper presents a new test to measure multitask accuracy in text models, highlighting the need for substantial improvements in achieving expert-level accuracy and addressing lopsided performance and low accuracy on socially important subjects.
- **[BigBench-Hard - Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models](https://arxiv.org/pdf/2206.04615.pdf)** - The paper introduces BIG-bench, a benchmark for evaluating language models on challenging tasks, providing insights on scale, calibration, and social bias.

## **Parameter- efficient fine tuning (PEFT)**

- **[Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf)** - This paper provides a systematic overview of Parameter-Efficient Fine-tuning (PEFT) Methods in all three categories discussed in the lecture videos.
- **[On the Effectiveness of Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2211.15583.pdf)** - The paper analyzes sparse fine-tuning methods for pre-trained models in NLP.

## **LoRA**

- **[LoRA Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)**  This paper proposes a parameter-efficient fine-tuning method that makes use of low-rank decomposition matrices to reduce the number of trainable parameters needed for fine-tuning language models.
- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)** - This paper introduces an efficient method for fine-tuning large language models on a single GPU, based on quantization, achieving impressive results on benchmark tests.

## **Prompt tuning with soft prompts**

- **[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)** - The paper explores "prompt tuning," a method for conditioning language models with learned soft prompts, achieving competitive performance compared to full fine-tuning and enabling model reuse for many tasks.