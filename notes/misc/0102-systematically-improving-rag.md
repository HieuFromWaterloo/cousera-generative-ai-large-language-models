# Systematically Improving Your RAG

- date: 2024-05-23
- source: https://jxnl.co/writing/2024/05/22/systematically-improving-your-rag/

--

### Summary of Lecture Notes on Systematic Improvement of RAG Systems

#### **Introduction**
- **Objective**: Provide a systematic approach to improving Retrieval-Augmented Generation (RAG) applications for companies.
- **Goal**: Enhance precision and recall, optimize retrieval, capture useful user feedback, and continuously monitor and evaluate the system.

#### **Key Areas of Improvement**

1. **Create Synthetic Questions and Answers**
   - **Purpose**: Quickly evaluate the system's precision and recall.
   - **Steps**:
     1. Generate synthetic questions for each text chunk in your database.
     2. Test the retrieval system using these synthetic questions.
     3. Calculate precision and recall scores to establish a baseline.
     4. Identify areas for improvement based on the baseline scores.
   - **Example**: For each document, create a set of questions it answers. Test if the system retrieves the correct document when these questions are asked.

2. **Combine Full-Text Search and Vector Search**
   - **Objective**: Achieve optimal retrieval performance by using both search methods.
   - **Observation**: Full-text search and embeddings often perform similarly, but full-text search can be significantly faster in some cases.
   - **Example**: In experiments with essays, both search methods performed similarly, but full-text search was 10 times faster. However, in a repository issue retrieval task, full-text search had a 55% recall, while embedding search had a 65% recall.

3. **Implement User Feedback Mechanisms**
   - **Purpose**: Capture specific feedback to study and improve system performance.
   - **Steps**:
     1. Collect user feedback on retrieved results.
     2. Use this feedback to identify and address specific issues in the retrieval process.
   - **Example**: Implement a feedback form where users can rate the relevance of retrieved documents and suggest improvements.

4. **Use Clustering to Identify Query Segments**
   - **Objective**: Find segments of queries with issues by breaking them down into topics and capabilities.
   - **Steps**:
     1. Cluster similar queries to identify common problem areas.
     2. Analyze these clusters to understand the types of issues and their causes.
   - **Example**: Cluster customer support queries into topics like "billing issues" and "technical support" to identify common retrieval problems within each topic.

5. **Build Specific Systems to Improve Capabilities**
   - **Purpose**: Develop targeted systems to address specific weaknesses identified through clustering and feedback.
   - **Steps**:
     1. Identify specific capabilities that need improvement.
     2. Build and fine-tune systems to enhance these capabilities.
   - **Example**: Develop a specialized retrieval system for technical support queries that incorporates domain-specific knowledge and terminologies.

6. **Continuous Monitoring and Evaluation**
   - **Objective**: Maintain and improve system performance as real-world data evolves.
   - **Steps**:
     1. Continuously monitor system performance using key metrics.
     2. Regularly evaluate and update the system based on new data and feedback.
   - **Example**: Set up automated monitoring tools to track precision and recall over time, and adjust the retrieval algorithms as necessary.

#### **Conclusion**
- **Complexity Management**: Break down complex RAG workflows into smaller, manageable pieces.
- **Systematic Approach**: Start with simple synthetic data to establish baselines, combine search methods, implement feedback loops, use clustering for targeted improvements, and continuously monitor and fine-tune the system.
- **Outcome**: By following this structured approach, you can incrementally enhance the performance and utility of RAG systems, unlocking their full potential to deliver exceptional user experiences and drive business value.

#### **Example of Systematic Improvement Process**

1. **Synthetic Data Creation**:
   - Generate questions like "What are secure coding practices?" for documents on security.
   - Test retrieval of these documents using both full-text and vector searches.
   - Calculate initial precision and recall (e.g., 97% recall with synthetic data).

2. **Combining Search Methods**:
   - Compare full-text search and vector search performance on different datasets.
   - Use the faster method where both perform similarly, or the more accurate one where there's a significant difference.

3. **User Feedback Implementation**:
   - Collect feedback on retrieved results for further analysis.
   - Adjust retrieval methods based on user feedback.

4. **Clustering Queries**:
   - Cluster queries into topics such as "security practices" and "coding standards."
   - Analyze clusters to identify common retrieval issues.

5. **Building Targeted Systems**:
   - Develop specialized retrieval systems for different topics based on identified issues.
   - Fine-tune these systems for improved accuracy and relevance.

6. **Continuous Monitoring**:
   - Set up monitoring tools to track system performance.
   - Regularly update and refine retrieval algorithms based on new data and feedback.

By implementing this systematic approach, you can significantly improve the performance and reliability of RAG systems.

### Summary of Lecture Notes on Utilizing Metadata and Search Techniques for RAG Systems

#### **Introduction**
- **Objective**: Improve search results by effectively utilizing metadata and combining full-text and vector search methods.
- **Goal**: Enhance the precision and recall of search queries through metadata extraction and optimized search techniques.

#### **Utilize Metadata**

**Steps to Incorporate Metadata**:
1. **Extract Relevant Metadata**:
   - Ensure metadata like date ranges, file names, and ownership is extracted from documents.
   - **Example**: Extracting publication dates, authors, and topics from research papers.

2. **Include Metadata in Search Indexes**:
   - Integrate extracted metadata into the search indexes to make it searchable.
   - **Example**: Indexing publication dates and authors alongside the main content.

3. **Query Understanding for Metadata Extraction**:
   - Develop mechanisms to extract metadata from user queries.
   - **Example**: Interpreting a query like "What are the latest developments?" to extract the date range and relevant terms.

4. **Expand Search Queries with Metadata**:
   - Enhance search queries by including relevant metadata to improve accuracy.
   - **Example**: Expanding "What are recent developments in AI?" to include a specific date range and trusted sources.

**Benefits**:
- **Improved Search Relevance**: Helps in capturing queries that cannot be fully addressed by text or semantic search alone.
- **Example in Practice**: 
  - Query: "What are recent developments in AI?"
  - Expanded Query: Includes terms like "recent," specific date ranges, and filters to trusted sources.

#### **Use Both Full-Text Search and Vector Search**

**Implementation Steps**:
1. **Implement Both Search Methods**:
   - Use both full-text search for speed and vector search for better recall.
   - **Example**: Full-text search for exact keyword matches; vector search for semantic relevance.

2. **Test Performance**:
   - Evaluate the effectiveness of each search method for your specific use case.
   - **Example**: Testing both methods on retrieving documents related to secure coding practices.

3. **Use a Single Database System**:
   - Store both full-text and vector data in a single system to avoid synchronization issues.
   - **Example**: A database system that supports full-text search, embeddings, and SQL queries.

4. **Evaluate Trade-offs**:
   - Balance speed and recall to find the optimal configuration for your application.
   - **Example**: Choosing full-text search for quick lookups and vector search for comprehensive query understanding.

**Benefits**:
- **Efficiency and Recall**: Combining both methods ensures quick and relevant search results.
- **Practical Example**:
  - Scenario: Construction data retrieval.
  - Problem: Separate indices for different projects causing synchronization issues.
  - Solution: Use a single system for full-text, embedding, and SQL queries to maintain consistency and accuracy.

---

### Key Takeaways

1. **Utilize Metadata**:
   - Extract and index relevant metadata from documents.
   - Implement query understanding to include metadata in search queries.
   - Expand search queries with metadata to improve precision and recall.

2. **Combine Full-Text and Vector Search**:
   - Use both methods to leverage their strengths (speed and recall).
   - Test and evaluate the performance of each method for your specific needs.
   - Consider a single database system to store and manage both types of data to avoid synchronization issues.

3. **Example Application**:
   - **Metadata Utilization**:
     - Query: "What are the latest AI trends?"
     - Expanded Query: "Recent AI trends" with date ranges and trusted sources.
   - **Search Combination**:
     - Implement full-text search for quick retrieval of exact matches.
     - Use vector search to capture semantic relevance and context.

By systematically incorporating metadata and combining full-text and vector search methods, you can significantly enhance the search capabilities and overall performance of your RAG systems, leading to more accurate and relevant results.

### Summary of Lecture Notes on Implementing Clear User Feedback Mechanisms

#### **Introduction**
- **Objective**: Gather data on system performance and identify areas for improvement through clear user feedback mechanisms.

#### **Steps to Implement User Feedback Mechanisms**

1. **Add Feedback Mechanisms**:
   - Incorporate user feedback options like thumbs up/down into your application.
   - **Example**: Adding a "Was this answer helpful? Yes/No" prompt after each response.

2. **Clear and Specific Copy**:
   - Ensure the feedback questions are specific and clearly describe what is being measured.
   - Avoid general questions and focus on specific aspects of the system's performance.
   - **Example**: Instead of asking "How did we do?", ask "Did we answer the question correctly?"

3. **Ask Specific Questions**:
   - Formulate feedback questions that target particular issues.
   - **Example**: "Was the information accurate?" or "Was the response time acceptable?"

4. **Use Feedback Data Effectively**:
   - Analyze the feedback data to identify areas needing improvement.
   - Prioritize fixes based on the specific issues highlighted by user feedback.
   - **Example**: If feedback indicates slow response times, prioritize optimizing system latency.

#### **Importance of Early Implementation**
- **Immediate Value**: Building feedback mechanisms early helps in quickly gathering useful data for evaluation.
- **Example**: Implementing feedback prompts in the initial release of a customer support chatbot to gather immediate insights on performance.

#### **Challenges and Solutions**
- **Confounding Variables**: General feedback like thumbs down can be ambiguous due to multiple factors (e.g., tone, latency).
- **Solution**:
  - Change the feedback question to be more precise.
  - Example: Instead of "Was this helpful?", ask "Did we answer your question correctly?"

#### **Example Process**

**Feedback Implementation**:
1. **Add Mechanism**: Implement a thumbs up/down option after each interaction.
2. **Clear Copy**: Use the question "Did we answer the question correctly? Yes or No."
3. **Specific Queries**: Include additional specific feedback options like "Was the response timely?" and "Was the tone appropriate?"

**Data Analysis**:
1. **Collect Data**: Gather responses from the feedback mechanism.
2. **Identify Issues**: Determine common areas of dissatisfaction (e.g., accuracy, response time).
3. **Prioritize Fixes**: Address the most frequent issues based on feedback (e.g., improving accuracy before focusing on tone).

---

### Key Takeaways

1. **Feedback Mechanisms**:
   - Implement user feedback options (e.g., thumbs up/down) in your application.
   - Use specific, clear questions to gather precise feedback.

2. **Specific Copy**:
   - Ensure feedback questions clearly describe the issue being measured.
   - Avoid general questions to reduce ambiguity.

3. **Data Utilization**:
   - Analyze feedback data to identify and prioritize areas for improvement.
   - Focus on specific issues highlighted by users (e.g., accuracy, latency).

4. **Example**:
   - Implementing a feedback system in a customer support application.
   - Asking "Did we answer your question correctly?" to gather precise feedback.
   - Analyzing data to prioritize improvements in response accuracy and speed.

By following these steps and focusing on clear, specific feedback mechanisms, you can effectively gather valuable data on system performance, identify precise areas for improvement, and enhance the overall user experience.

### Summary of Lecture Notes on Clustering and Modeling Topics

#### **Introduction**
- **Objective**: Analyze user queries and feedback to identify topic clusters, capabilities, and areas of user dissatisfaction for prioritizing improvements.

#### **Why Clustering and Modeling Topics?**
- **Example Case**:
  - A company with a technical documentation search system used clustering to identify key issues:
    - **Topic Clusters**: Many queries were about a recently updated product feature, but the system was not retrieving the latest documentation, causing user frustration.
    - **Capability Gaps**: Users frequently asked for troubleshooting steps and error code explanations, but the system struggled to provide direct, actionable answers.

- **Outcome**:
  - **Improvements**: Updated documentation for the product feature and implemented a feature to extract step-by-step instructions and error code explanations.
  - **Result**: Higher user satisfaction and reduced support requests.

#### **Identifying Patterns**
- **Topic Clusters**:
  - **Purpose**: Determine if users are asking about specific topics more than others.
  - **Action**: Focus on creating more content in those areas or improving retrieval of existing content.
  - **Example**: Frequently asked questions about a new software feature.

- **Capabilities**:
  - **Purpose**: Identify types of questions your system cannot answer.
  - **Action**: Develop new features or capabilities to address these gaps, such as direct answer extraction or domain-specific reasoning.
  - **Example**: System struggles with providing direct troubleshooting steps or explanations for error codes.

#### **Continuous Analysis for Improvement**
- **Process**:
  - Regularly analyze user queries and feedback to identify high-impact areas.
  - Allocate resources effectively based on data-driven insights.
  - **Example**: Prioritize updates and new features based on the most critical user issues.

- **Collaboration with Domain Experts**:
  - **Step**: Work with domain experts to explicitly define categories for topic clusters and capabilities.
  - **Outcome**: Build systems to tag data as it comes in, similar to how ChatGPT generates automatic titles for conversations.

- **Integration with Tools**:
  - **Tools**: Use tools like Amplitude or Sentry to track and analyze the types of queries people are asking.
  - **Purpose**: Understand query patterns to prioritize capabilities and topics for improvement.
  - **Example**: Implement a tracking system to monitor queries related to different capabilities such as fetching documents, images, or performing comparisons.

#### **Example Implementation**

1. **Identify Topic Clusters**:
   - **Method**: Analyze user queries to find common topics.
   - **Example**: Queries related to "secure coding practices" or "error code explanations."

2. **Identify Capability Gaps**:
   - **Method**: Determine which types of questions the system struggles to answer.
   - **Example**: Difficulty in providing troubleshooting steps or fetching specific documents.

3. **Collaborate with Experts**:
   - **Action**: Discuss identified clusters and gaps with domain experts to categorize them explicitly.
   - **Example**: Define categories such as "ownership and responsibility," "fetching tables," "no synthesis."

4. **Implement Tagging System**:
   - **Action**: Tag incoming data with the identified topics and capabilities.
   - **Example**: Automatically classify queries and track them using tools like Amplitude.

5. **Continuous Monitoring**:
   - **Action**: Use analytics tools to monitor the types of queries and their frequency.
   - **Example**: Track how often users ask for "secure coding practices" and prioritize related content updates.

#### **Conclusion**
- **Data-Driven Prioritization**: Continuously analyze topic clusters and capability gaps to identify high-impact areas for improvement.
- **Effective Resource Allocation**: Allocate resources based on the most critical user issues identified through clustering and modeling.
- **Improvement Strategy**: Regularly update and enhance the system based on user feedback and query analysis to improve user satisfaction and reduce support requests.

By implementing this structured approach to clustering and modeling topics, you can systematically improve the performance and utility of your RAG systems, ensuring they meet user needs more effectively and efficiently.

### Summary of Lecture Notes on Continuously Monitoring and Experimenting for RAG Systems

#### **Introduction**
- **Objective**: Continuously monitor system performance and run experiments to test and implement improvements.
- **Goal**: Ensure ongoing enhancements in precision, recall, and overall system effectiveness.

#### **Steps for Continuous Monitoring and Experimentation**

1. **Set Up Monitoring and Logging**:
   - Track system performance over time through comprehensive monitoring and logging.
   - **Example**: Implement logging to capture metrics like response time, precision, recall, and user satisfaction ratings.

2. **Regularly Review Data**:
   - Analyze collected data to identify trends, patterns, and emerging issues.
   - **Example**: Monthly reviews of performance logs to spot decreases in precision or increases in user dissatisfaction.

3. **Design and Run Experiments**:
   - Develop experiments to test potential improvements in the system.
   - **Example**: Experiment with different search parameters, metadata inclusion, or embedding models to enhance retrieval accuracy.

4. **Measure Impact**:
   - Evaluate the effects of changes on key metrics such as precision, recall, and user satisfaction.
   - **Example**: Compare performance metrics before and after implementing a new embedding model to assess its impact.

5. **Implement Effective Changes**:
   - Apply changes that demonstrate significant improvements based on experimental results.
   - **Example**: Integrate a new search algorithm that increases recall without compromising precision.

#### **Systematic Improvement Process**

1. **Utilize Synthetic and User Data**:
   - Use synthetic questions and user data with ratings to inform the improvement process.
   - **Example**: Analyze synthetic data sets to establish baselines and user ratings to identify areas of dissatisfaction.

2. **Topic Modeling and Clustering**:
   - Perform topic modeling on user questions and correlate with thumbs up/down ratings to identify underperforming clusters.
   - **Example**: Identify that questions about "deadlines" receive low satisfaction ratings and need improvement.

3. **Cadence of Analysis**:
   - Regularly analyze clusters to determine user dissatisfaction levels and prioritize specific use cases for improvement.
   - **Example**: Weekly analysis to update focus areas based on user feedback trends.

4. **Adapt to Changes**:
   - Adjust to new distributions when onboarding new organizations or encountering different use cases.
   - **Example**: Onboard a new client whose primary concern is "deadlines," shifting focus to improving deadline-related queries.

#### **Example Implementation**

1. **Monitoring Setup**:
   - Implement logging for response time, precision, recall, and user ratings.
   - **Example**: Use tools like Elasticsearch and Kibana to visualize performance metrics.

2. **Data Review**:
   - Monthly data analysis to identify declining performance in specific areas.
   - **Example**: Detect a drop in precision for queries related to "error codes."

3. **Experiment Design**:
   - Test the addition of metadata in search queries to improve accuracy.
   - **Example**: Experiment with adding publication dates to improve retrieval relevance.

4. **Impact Measurement**:
   - Evaluate changes in precision and recall after experiments.
   - **Example**: Measure the improvement in recall after integrating a new embedding model.

5. **Implementation of Changes**:
   - Apply changes that improve key metrics.
   - **Example**: Deploy a new search parameter configuration that boosts precision.

#### **Continuous Adaptation and Improvement**

1. **Topic Modeling and Clustering**:
   - Use clustering to identify areas needing improvement based on user feedback.
   - **Example**: Recognize that "troubleshooting" queries are underperforming and prioritize enhancements.

2. **Regular Updates**:
   - Continuously update the system based on real-world data and user feedback.
   - **Example**: Adapt to a new client's needs by focusing on deadline-related query improvements.

---

### Key Takeaways

1. **Continuous Monitoring**:
   - Set up robust monitoring and logging to track system performance.
   - Regularly review data to identify trends and issues.

2. **Experimentation**:
   - Design and run experiments to test potential improvements.
   - Measure the impact of changes on precision, recall, and user satisfaction.

3. **Systematic Improvement**:
   - Use synthetic data and user feedback to drive improvements.
   - Apply topic modeling and clustering to identify underperforming areas.

4. **Adaptability**:
   - Continuously adapt to new data and changing user needs.
   - Prioritize improvements based on user feedback and real-world use cases.

By following these steps, you can systematically enhance the performance and utility of your RAG systems, ensuring they remain effective and relevant over time.

### Summary of Lecture Notes on Balancing Latency and Performance

#### **Introduction**
- **Objective**: Make informed decisions about trade-offs between system latency and search performance based on specific use cases and user requirements.

#### **Steps to Balance Latency and Performance**

1. **Understand Requirements**:
   - **Latency Requirements**: Determine acceptable response times for your application.
   - **Performance Requirements**: Define the desired level of search accuracy (precision and recall).
   - **Example**: A medical diagnostic tool may prioritize high recall over latency, while a general search tool may prioritize low latency.

2. **Measure Impact of Configurations**:
   - Test different system configurations to see how they affect latency and performance.
   - Use synthetic questions to evaluate the impact of changes on recall and latency.
   - **Example**: Run queries with and without specific features (like a parent document retriever) to measure changes in recall and latency.

3. **Make Trade-Offs**:
   - Evaluate the trade-offs based on what's most important for your users.
   - Consider the specific requirements of different use cases.
   - **Example**: For a medical diagnostic tool, a 1% improvement in recall might justify a 20% increase in latency. For a documentation search, a 20% latency increase might not be acceptable if recall only improves by 1%.

#### **Example Process**

1. **Setup and Measure**:
   - Implement synthetic questions to test against different configurations.
   - Measure recall and latency for each configuration.
   - **Example**: Testing a new retrieval feature that increases recall but also adds latency.

2. **Evaluate Trade-Offs**:
   - Compare the results to understand the impact on user experience.
   - **Scenario 1**: If recall doubles and latency increases by 20%, consider if the increased recall is worth the higher latency.
   - **Scenario 2**: If recall increases by 1% and latency doubles, decide if the minimal performance gain justifies the significant increase in latency.

3. **Context-Based Decision Making**:
   - **High-Stakes Applications**: In applications like medical diagnostics, prioritize recall even if latency increases slightly.
   - **General Applications**: In general search tools, prioritize faster results if recall improvements are marginal.
   - **Example**: A 1% improvement in recall may be critical for medical diagnostics but not for a documentation search tool.

#### **Conclusion**

**Key Considerations**:
- **Application Context**: Understand the context and criticality of your application.
- **User Impact**: Consider how changes in latency and performance will affect user satisfaction and system effectiveness.
- **Continuous Evaluation**: Regularly test and measure the impact of different configurations to make informed decisions.

**Example**:
- **Medical Diagnostic Tool**: A slight increase in latency for a significant gain in recall can be justified due to the high stakes.
- **General Search Tool**: Faster response times may be prioritized over small gains in recall to enhance user experience.

By following these steps and focusing on the specific needs of your application, you can effectively balance latency and performance, ensuring your RAG system meets user expectations and delivers optimal results.