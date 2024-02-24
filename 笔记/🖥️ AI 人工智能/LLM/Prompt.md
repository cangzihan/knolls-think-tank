---
tags:
  - GPT4
  - Prompt
  - OpenAI
  - 大模型
---

# Prompt
- OpenAI官方文档: https://platform.openai.com/docs/guides/prompt-engineering

## Six strategies for getting better results
1. Write clear instructions
2. Provide reference text
3. Split complex tasks into simpler subtasks
4. Give the model time to "think"
5. Use external tools
6. Test changes systematically

### Write clear instructions

If outputs are too long, ask for brief replies. If outputs are too simple, ask for expert-level writing. If you dislike the format, demonstrate the format you’d like to see. The less the model has to guess at what you want, the more likely you’ll get it.

Tactics:
- Include details in your query to get more relevant answers

| Worse | Better |
|:---:|:---|
| How do I add numbers in Excel?   | How do I add up a row of dollar amounts in Excel? I want to do this automatically for a whole sheet of rows with all the |
| | totals ending up on the right in a column called "Total".   |
| Who's president?  |	Who was the president of Mexico in 2021, and how frequently are elections held?  |
|Write code to calculate the Fibonacci sequence.	| Write a TypeScript function to efficiently calculate the Fibonacci sequence. Comment the code liberally to explain |
| | what each piece does and why it's written that way.|
|Summarize the meeting notes.  |	Summarize the meeting notes in a single paragraph. Then write a markdown list of the speakers and each of their |
| |  key points. Finally, list the next steps or action items suggested by the speakers, if any.|

- Ask the model to adopt a persona
```
{
  "SYSTEM": "When I ask for help to write something, you will reply with a document that contains at least one joke or playful comment in every paragraph."
  "USER": "Write a thank you note to my steel bolt vendor for getting the delivery in on time and in short notice. This made it possible for us to deliver an important order."
}
```

- Use delimiters to clearly indicate distinct parts of the input

Delimiters like triple quotation marks, XML tags, section titles, etc. can help demarcate sections of text to be treated differently.
```
USER
  Summarize the text delimited by triple quotes with a haiku.

  """insert text here"""
```

```
SYSTEM
  You will be provided with a pair of articles (delimited with XML tags) about the same topic. First summarize the arguments of each article. Then indicate which of them makes a better argument and explain why.
USER
  <article> insert first article here </article>

  <article> insert second article here </article>
```
- Specify the steps required to complete a task
```
SYSTEM
  Use the following step-by-step instructions to respond to user inputs.

  Step 1 - The user will provide you with text in triple quotes. Summarize this text in one sentence with a prefix that says "Summary: ".

  Step 2 - Translate the summary from Step 1 into Spanish, with a prefix that says "Translation: ".

USER
  """insert text here"""
```
- Provide examples

  Providing general instructions that apply to all examples is generally more efficient than demonstrating all permutations of a task by example, but in some cases providing examples may be easier. For example, if you intend for the model to copy a particular style of responding to user queries which is difficult to describe explicitly. This is known as "few-shot" prompting.
```
SYSTEM
  Answer in a consistent style.
USER
  Teach me about patience.
ASSISTANT
  The river that carves the deepest valley flows from a modest spring; the grandest symphony originates from a single note; the most intricate tapestry begins with a solitary thread.
USER
  Teach me about the ocean.
```

- Specify the desired length of the output
```
Summarize the text delimited by triple quotes in about 50 words.
Summarize the text delimited by triple quotes in 2 paragraphs.
Summarize the text delimited by triple quotes in 3 bullet points.
```

### Provide reference text
```
SYSTEM
  Use the provided articles delimited by triple quotes to answer questions. If the answer cannot be found in the articles, write "I could not find an answer."
USER
  <insert articles, each delimited by triple quotes>

  Question: <insert question here>
```

### Split complex tasks into simpler subtasks

- Use intent classification to identify the most relevant instructions for a user query

  限定查询范围降低错误率。 This can also result in lower costs since larger prompts cost more to run ([see pricing information](https://openai.com/pricing)).

```
SYSTEM
  You will be provided with customer service queries. Classify each query into a primary category and a secondary category. Provide your output in json format with the keys: primary and secondary.

  Primary categories: Billing, Technical Support, Account Management, or General Inquiry.

  Billing secondary categories:
    - Unsubscribe or upgrade
    - Add a payment method
    - Explanation for charge
    - Dispute a charge

    Technical Support secondary categories:
    - Troubleshooting
    - Device compatibility
    - Software updates

    Account Management secondary categories:
    - Password reset
    - Update personal information
    - Close account
    - Account security

    General Inquiry secondary categories:
    - Product information
    - Pricing
    - Feedback
    - Speak to a human
USER
  I need to get my internet working again.

```

Based on the classification of the customer query, a set of more specific instructions can be provided to a model for it to handle next steps. For example, suppose the customer requires help with "troubleshooting".
```
SYSTEM
  You will be provided with customer service inquiries that require troubleshooting in a technical support context. Help the user by:

  - Ask them to check that all cables to/from the router are connected. Note that it is common for cables to come loose over time.
  - If all cables are connected and the issue persists, ask them which router model they are using
  - Now you will advise them how to restart their device:
  -- If the model number is MTD-327J, advise them to push the red button and hold it for 5 seconds, then wait 5 minutes before testing the connection.
  -- If the model number is MTD-327S, advise them to unplug and replug it, then wait 5 minutes before testing the connection.
  - If the customer's issue persists after restarting the device and waiting 5 minutes, connect them to IT support by outputting {"IT support requested"}.
  - If the user starts asking questions that are unrelated to this topic then confirm if they would like to end the current chat about troubleshooting and classify their request according to the following scheme:

  <insert primary/secondary classification scheme from above here>
USER
  I need to get my internet working again.

```

- For dialogue applications that require very long conversations, summarize or filter previous dialogue

  **Q**: 对话变多时，输入的长度会变长，而输入长度不可能无限增加下去.

  There are various workarounds to this problem, one of which is to summarize previous turns in the conversation. Once the size of the input reaches a predetermined threshold length, this could trigger a query that summarizes part of the conversation and the summary of the prior conversation could be included as part of the system message. Alternatively, prior conversation could be summarized asynchronously in the background throughout the entire conversation.

  An alternative solution is to dynamically select previous parts of the conversation that are most relevant to the current query. See the tactic "Use embeddings-based search to implement efficient knowledge retrieval".

- Summarize long documents piecewise and construct a full summary recursively
  **Q**: 还是输入长度不可能无限增加下去的问题

   Section summaries can be concatenated and summarized producing summaries of summaries. This process can proceed recursively until an entire document is summarized. If it’s necessary to use information about earlier sections in order to make sense of later sections, then a further trick that can be useful is to include a running summary of the text that precedes any given point in the book while summarizing content at that point. The effectiveness of this procedure for summarizing books has been studied in previous [research](https://openai.com/research/summarizing-books) by OpenAI using variants of GPT-3.
