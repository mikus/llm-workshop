#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";

import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";

import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";

import {
    RunnableParallel,
    RunnablePassthrough,
  } from "@langchain/core/runnables";
  
const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
});

const parser = new StringOutputParser();

var chatParser = chat.pipe(parser);

// global storage for intermediate responses
var responses = {};

// helper function to store intermediate responses
const inspector = (response) => {
  Object.assign(responses, response);
  return response
};

const store = new Map();

const getMessageHistory = (sessionId) => {
  if (store.has(sessionId)) {
    return store.get(sessionId);
  } else {
    const newChatMessageHistory = new ChatMessageHistory();
    store.set(sessionId, newChatMessageHistory);
    return newChatMessageHistory;
  }
};

// Prompt template for the game_show_host_chain
const gameShowHostPrompt = ChatPromptTemplate.fromMessages([
    ["system", `You are a game show host. Given a topic, write a question for the contestants. \
                Do NOT write anything besides the question. Do NOT repeat questions from the history \
                and do NOT ask similar questions as you already did. Analyze step by step the questions you already \
                asked in history and create anothere one which is quite different from the previous ones.
                Topic: {input}
                Question:`],
    ["placeholder", "{chat_history}"],
    ["human", "{input}"],
]);

// Prompt template for the contestant_one_chain
const contestantOnePrompt = PromptTemplate.fromTemplate(`
You are a contestant for a game show. Given a question, answer the question.

Question: {question}

Answer:
`);

// Prompt template for the contestant_two_chain
// The prompt is intended to make contestant two answer incorrectly to test the judge chain.
const contestantTwoPrompt = PromptTemplate.fromTemplate(`
You are a contestant for a game show. \
You know very little about general topics. \
Your answer to the given question is incorrect and irrelevant. \
Even if you know the answer, make it funny.
Always answer in master yoda style.

Question: {question}

Answer:
`);

// Prompt template for the judge_chain
const judgePrompt = PromptTemplate.fromTemplate(`
You the judge of a contest. Given a question and two answers: Answer 1 and Answer 2, you should pick the correct answer.\
However, both answers might be correct or incorrect at the same time. \
If you don't know the answer to a question you should admit that you don't know.

Question:{question}

Answer 1: {answer1}
Answer 2: {answer2}

The correct answer is:
`);

// Define chains for each prompt template
const gameShowHostChain = gameShowHostPrompt.pipe(chatParser).pipe({question: new RunnablePassthrough()}).pipe(inspector);
const gameShowHostChainWithHistory = new RunnableWithMessageHistory({
    runnable: gameShowHostChain,
    getMessageHistory: getMessageHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    config: {configurable: {sessionId: "game123"}},
});
const contestantOneChain = contestantOnePrompt.pipe(chatParser).pipe({answer1: new RunnablePassthrough()}).pipe(inspector);
const contestantTwoChain = contestantTwoPrompt.pipe(chatParser).pipe({answer2: new RunnablePassthrough()}).pipe(inspector);
const judgeChain = judgePrompt.pipe(chatParser).pipe({correctAnswer: new RunnablePassthrough()}).pipe(inspector);


// Combine the four chains
const gameShowChain = gameShowHostChainWithHistory
                        .pipe({
                                answer1: contestantOneChain,
                                answer2: contestantTwoChain,
                                question: new RunnablePassthrough(),
                            })
                        .pipe(judgeChain);

// Run the chain
var gameShowResponse = await gameShowChain.invoke({input: "eurovision song contest"});
console.log(gameShowResponse);
