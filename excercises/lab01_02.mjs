#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";

const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
});

/*
## Prompt Templates

The input to the LLM (or any foundational model) is called a prompt. Prompts are typically in the form of text. It provides the LLM with all the necessary information to produce the respective response.
Prompt templates are parameterized model inputs serving as pre-defined recipes for LLMs. These templates can be reusable and enables LLMs to adapt to more number of tasks with minimal effort.
*/

/*
### String PromptTemplates
These prompt templates are used to format a single string, and generally are used for simpler inputs. For example, a common way to construct and use a PromptTemplate is as follows:
*/

const promptTemplate = PromptTemplate.fromTemplate(
    "Tell me a joke about {topic}"
);

var template = await promptTemplate.invoke({ topic: "cats" });
console.log(template);

var response = await chat.invoke(template);
console.log(response.content);


/*
### Variable Number of Inputs

We can use multiple inputs while defining a prompt template.
Note that when using multiple inputs, the input keys should match the keys in the prompt template.
*/


const multiInputPromptTemplate = PromptTemplate.fromTemplate(
"Tell me a {content_type} about a {topic}."
);

template = await multiInputPromptTemplate.invoke({ content_type: "bedtime story", topic: "cat" });
console.log(template);

response = await chat.invoke(template);
console.log(response.content);
