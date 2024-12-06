#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";

import {
    AIMessage,
    HumanMessage,
    SystemMessage,
    trimMessages,
} from "@langchain/core/messages";
  
const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
});

var trimmer = trimMessages({
    maxTokens: 10,
    strategy: "last",
    tokenCounter: chat,
    includeSystem: true,
    allowPartial: false,
    startOn: "human"
});

var messages = [
    new SystemMessage("you're a good assistant"),
    new HumanMessage("hi! I'm bob"),
    new AIMessage("hi!"),
    new HumanMessage("I like vanilla ice cream"),
    new AIMessage("nice"),
    new HumanMessage("whats 2 + 2"),
    new AIMessage("4"),
    new HumanMessage("thanks"),
    new AIMessage("no problem!"),
    new HumanMessage("having fun?"),
    new AIMessage("yes!"),
];

console.log(await trimmer.invoke(messages));

var chain = trimmer.pipe(chat).pipe(parser);

var chatHistory = new ChatMessageHistory([...messages]);

var chainWithHistory = new RunnableWithMessageHistory({
  runnable: chain,
  getMessageHistory: (sessionId) => {
    if (sessionId !== "1") {
      throw new Error("Session not found");
    }
    return chatHistory;
  },
});

var response = await chainWithHistory.invoke(
  [new HumanMessage("what's my name?")],
  { configurable: { sessionId: "1" } }
);

console.log(response);

var chatHistory = new ChatMessageHistory([...messages]);

var response = await chainWithHistory.invoke(
  [new HumanMessage("what math problem did i ask?")],
  { configurable: { sessionId: "1" } }
);

console.log(response);
