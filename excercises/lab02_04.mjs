#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate, FewShotPromptTemplate } from "@langchain/core/prompts";


const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4-turbo",
});


/*
### 4. In-context learning

As pre-trained large language models learn from large and diverse data sources, they tend to build a holistic view of languages and text. This advantage allows them to learn from some input-output pairs that they are presented within the input texts. 

In this section, we will explain this __"in-context"__ learning capability with some examples. Depending on the level of information presented to the model, we can use zero-shot, one-shot or few-shot learning. We start with the most extreme case, no information presented to the model. This is called __"zero-shot-learning"__.
*/

/*
#### Zero-shot learning:
Assume the model is given a translation task and an input word.
 */

var template = PromptTemplate.fromTemplate(
    `Translate the following word from English to Spanish 
    word: {word} 
    translation: `
);
var prompt = await template.invoke({ word: "cat" });

response = await chat.invoke(prompt);
console.log(response.content);


/*
#### One-shot learning:
We can give the model one example and let it learn from the example to solve a problem. Below, we provide an example sentence about a cat and the model completes the second sentence about a table in a similar way.
*/

template = PromptTemplate.fromTemplate(
    `Answer the last question
    question: what is a cat?
    answer: cat is an animal
    ##
    last question: what is a {word}?
    answer: {word} is `
);
prompt = await template.invoke({ word: "car" });

response = await chat.invoke(prompt);
console.log(response.content);


/*
#### Few-shot learning:
We can give the model multiple examples to learn from. Providing more examples can help the model produce more accurate results. Let's also change the style of the example answers by adding some __negation__ to them.
*/

const examplePrompt = PromptTemplate.fromTemplate(
    `Question: {question}
    Answer: {answer}`
);
  
const examples = [
    {
      question: "what is a car?",
      answer: "car is not an animal",
    },
    { 
      question: "what is a cat?",
      answer: "cat is not a vehicle",
    }
];
  
template = new FewShotPromptTemplate({
    examples,
    examplePrompt,
    suffix: "Question: {input}",
    inputVariables: ["input"],
    prefix: "Answer the last question",
});
  
var prompt = await template.format({
    input: "shoe is",
});
console.log(prompt.toString());

var response = await chat.invoke(prompt);
console.log(response.content);


/*
Let's try one more time. This time we remove the instruction and try to complete the last sentence.
*/

template = new FewShotPromptTemplate({
    examples,
    examplePrompt,
    suffix: "Question: {input}",
    inputVariables: ["input"],
});
  
var prompt = await template.format({
    input: "shoe is",
});
console.log(prompt.toString());

var response = await chat.invoke(prompt);
console.log(response.content);
