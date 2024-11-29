#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";


const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4-turbo",
});

/*
### 3. Text Generation

Text generation is one of the common use cases for Large Language Models. The main purpose is to generate some high quality text considering a given input. We will cover a few examples here.

__Customer service example:__

Let's start with a customer feedback example. Assume we want to write an email to a customer who had some problems with a product that they purchased.
*/


var prompt = "Write an email from Acme company customer service \
based on the following email that was received from a customer \
Customer email: \"I am not happy with this product. I had a difficult \
time setting it up correctly because the instructions do not cover all \
the details. Even after the correct setup, it stopped working after \
a week.\"";


var response = await chat.invoke(prompt);
console.log(response.content);

/*
Nice! The generated text asks customer to provide more details to resolve the issue.
*/


prompt = "Product: Sunglasses.  \
Keywords: polarized, style, comfort, UV protection. \
List three different product descriptions \
for the product listed above using \
at least two of the provided keywords."

response = await chat.invoke(prompt);
console.log(response.content);
