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

var chatParser = chat.pipe(parser);

// Physics template used for answering questions about physics
const physicsTemplate = `You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know. Be precise.

Here is a question:
{input}`

// Poem template used for writing poems as per asked
const poemTemplate = `You are a very good poet. You are skilled at writing short poems on a given topic.

Here is a statement:
{input}`

// History template for answering questions about historical events
const historyTemplate = `You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}`

// Provide a name, description and template to make it easier for the router chain to select the right prompt template
const promptInfos = [
    {
        name: "physics",
        description: "Good for answering questions about physics",
        promptTemplate: physicsTemplate,
    },
    {
        name: "poem",
        description: "Good for writing poems",
        promptTemplate: poemTemplate,
    },
    {
        name: "history",
        description: "Good for answering history questions",
        promptTemplate: historyTemplate,
    },
];

// Create a list of chains, one for each prompt template
const chainNames = promptInfos.map((info) => info["name"]);

const routingSchema = {
    title: "destination",
    description: "name of the destination chain",
    type: "object",
    properties: {
        destination: {
            type: "string",
            enum: chainNames,
        }
    },
    required: ["destination"]
};

var destinationChains = new Map();
for (let info of promptInfos) {
    var name = info["name"];
    var promptTemplate = info["promptTemplate"];
    var prompt = PromptTemplate.fromTemplate(promptTemplate);
    var chain = prompt.pipe(chatParser);
    destinationChains.set(name, chain);
};

const routeSystem = "Route the user's query to either the physics, poem, or history expert."
const routePrompt = ChatPromptTemplate.fromMessages([
        ("system", routeSystem),
        ("human", "{input}"),
]);

// Setting the default chain to a generic ConversationChain
const defaultChain = chatParser.pipe(new RunnablePassthrough());

const routerChain = routePrompt.pipe(chat.withStructuredOutput(routingSchema));

const route = (info) => {
    var name = info['route']['destination'];
    if (destinationChains.has(name)) {
        return destinationChains.get(name);
    }
    return defaultChain;
};

var chain = RunnablePassthrough.assign({route: routerChain}).pipe(route);

// Prompt about writing a poem. The verbose should mention the which chain was selected.
var routerResponse1 = await chain.invoke({input: "Compose a short poem about roses."});
console.log(routerResponse1);

console.log("------");

// Prompt about a physics topic. The verbose should mention the which chain was selected.
var routerResponse2 = await chain.invoke({input: "What are the three laws of thermodynamics?"});
console.log(routerResponse2);

console.log("------");

// Prompt about a historical event. The verbose should mention the which chain was selected.
var routerResponse3 = await chain.invoke({"input": "What events led to the second World War?"});
console.log(routerResponse3);
