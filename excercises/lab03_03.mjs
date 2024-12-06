#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";


import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { DataFrame } from 'pandas-js';


const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
});

const parser = new StringOutputParser();


// Here we use a global variable to store the chat message history.
// This will make it easier to inspect it to see the underlying results.
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



async function promptAndPrintMemory(prompts, chainWithHistory, history, inputKey, config) {
    // Store the responses
    let responses = [];

    // Repeatedly prompting the chain and observing the memory
    for (let prompt of prompts.values()) {
        let input = {};
        input[inputKey] = prompt;
        let response = await chainWithHistory.invoke(input, config);
        responses.push({
            input: prompt,
            history: history.messages.map((item) => item.content + " / "),
            response: response
        });
    }

    // Store and print the responses in a dataframe
    const df = new DataFrame(responses);
    console.log(df.toString());
}

var prompt = ChatPromptTemplate.fromMessages(
    [
        ("system", "You are a pirate. Answer the following questions as best you can."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
);

var chain = prompt.pipe(chat).pipe(parser);

var sessionId = "pirate123";

var chainWithHistory = new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: getMessageHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    config: {configurable: {sessionId: sessionId}}
});

// Sequence of prompts for the ConversationChain
var prompts = [
    "What is LangChain JS?",
    "Can it help with deploying ML models?",
    "Do you need any special environment to access it?",
];

// Use the helper function defined to sequentially prompt the chain and observe the memory buffer with each prompt
await promptAndPrintMemory(prompts, chainWithHistory, getMessageHistory(sessionId), 'input');
