#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";

import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";

import { StringOutputParser } from "@langchain/core/output_parsers";

import {
    RunnableParallel,
    RunnablePassthrough,
  } from "@langchain/core/runnables";
  
const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
});

const parser = new StringOutputParser();


// The first prompt for the SimpleSequentialChain
var contentWriterPrompt = PromptTemplate.fromTemplate(`
    You are a content writer. Write an article about the following topic.
    Topic: {topic}
    Article:
`);

// The second prompt for the SimpleSequentialChain
var childrenBookAuthorPrompt = PromptTemplate.fromTemplate(`
    Rewrite the following article such that a five year old can understand.
    Article: {article}
    Article that a five year old can understand:
`);

// Define chains for each prompt template
var contentWriterChain = contentWriterPrompt.pipe(chat).pipe(parser);
var childrenBookAuthorChain = childrenBookAuthorPrompt.pipe(chat).pipe(parser);

// Combine the two chains
var combinedChain = contentWriterChain.pipe({article: new RunnablePassthrough()}).pipe(childrenBookAuthorChain);

// Run the combined chain
var response = await combinedChain.invoke({topic: "global warming"});
console.log(response);
