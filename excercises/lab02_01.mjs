#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate, FewShotPromptTemplate } from "@langchain/core/prompts";


const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4-turbo",
});

/*
### 1. Text sumarization
With text summarization, the main purpose is to create a shorter version of a given text while preserving the relevant information in it. 

We use the following text from the **Capabilities** section of [OpeanAI o1 model from Wikipedia](https://en.wikipedia.org/wiki/OpenAI_o1).
*/

const text = "According to OpenAI, o1 has been trained using a new optimization \
algorithm and a dataset specifically tailored to it; while also meshing \
in reinforcement learning into its training. OpenAI described o1 as a \
complement to GPT-4o rather than a successor. \
o1 spends additional time thinking (generating a chain of thought) before \
generating an answer, which makes it more effective for complex reasoning \
tasks, particularly in science and mathematics. Compared to previous models, \
o1 has been trained to generate long \"chains of thought\" before returning \
a final answer. According to Mira Murati, this ability to think before \
responding represents a new, additional paradigm, which is improving model \
outputs by spending more computing power when generating the answer, whereas \
the model scaling paradigm improves outputs by increasing the model size, \
training data and training compute power. OpenAI's test results suggest \
a correlation between accuracy and the logarithm of the amount of compute \
spent thinking before answering. \
o1-preview performed approximately at a PhD level on benchmark tests related \
to physics, chemistry, and biology. On the American Invitational Mathematics \
Examination, it solved 83% (12.5/15) of the problems, compared to 13% (1.8/15) \
for GPT-4o. It also ranked in the 89th percentile in Codeforces coding competitions. \
o1-mini is faster and 80% cheaper than o1-preview. It is particularly suitable for \
programming and STEM-related tasks, but does not have the same \"broad world knowledge\" \
as o1-preview. \
OpenAI noted that o1's reasoning capabilities make it better at adhering to safety \
rules provided in the prompt's context window. OpenAI reported that during a test, \
one instance of o1-preview exploited a misconfiguration to succeed at a task that \
should have been infeasible due to a bug. OpenAI also granted early access to the UK \
and US AI Safety Institutes for research, evaluation, and testing. According to OpenAI's \
assessments, o1-preview and o1-mini crossed into \"medium risk\" in CBRN (biological, \
chemical, radiological, and nuclear) weapons. Dan Hendrycks wrote that \"The model \
already outperforms PhD scientists most of the time on answering questions related \
to bioweapons.\" He suggested that these concerning capabilities will continue to increase.";

/*
Let's start with the first summarization example below. We pass this text as well as the instruction to summarize it. The instruction part of the prompt becomes __Summarize it__
*/

var prompt = "The following is a text about OpenAI o1 model. Summarize it \
Text: " + text;

var response = await chat.invoke(prompt);
console.log(response.content);

/*
Nice. This text is shorter. We can set the desired lenght of the summary by adding more constraints to the instructions. Let's create a one-sentence summary of this text. The instruction part of the prompt becomes the following: __Summarize it in one sentence__
*/

prompt = "The following is a text about OpenAI o1 model. Summarize it in one sentence \
Text: " + text;

response = await chat.invoke(prompt);
console.log(response.content);
