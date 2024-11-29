#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import{ HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";

const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
});

/*
## Chat
Chat models are language models that use a sequence of messages as inputs and return messages as outputs (as opposed to using plain text). 

#### Chat Roles
Roles are used to distinguish between different types of messages in a conversation and help the chat model understand how to respond to a given sequence of messages.

* **system** - Used to tell the chat model how to behave and provide additional context. Not supported by all chat model providers.
* **user** - Represents input from a user interacting with the model, usually in the form of text or other interactive input.
* **assistant** - Represents a response from the model, which can include text or a request to invoke tools.
* **tool** - A message used to pass the results of a tool invocation back to the model after external data or processing has been retrieved. Used with chat models that support tool calling.
*/


var response = await chat.invoke([new HumanMessage("Hello, how are you?")]);
console.log(response.content);



response = await chat.invoke(
    [
        new SystemMessage("You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        new HumanMessage("I like tomatoes, what should I eat?"),
    ]
);
console.log(response.content);

/*
You can also pass more chat history w/ responses from the AI
*/

var nextResponse = await chat.invoke(
    [
        new SystemMessage("You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        new HumanMessage("I like tomatoes, what should I eat?"),
        new AIMessage(response.content),
        new HumanMessage("What are the ingredients required for making it?"),
    ]
);
console.log(nextResponse.content);

/*
### Prompt Templates for Chats

`ChatPromptTemplates` are designed for chat models. Conversational tasks typically involve a `system` message, `user` message and an `assistant` message.

In the following example, we would only use the `system` message and `user` message as this would be a one-off conversation.
*/


const chatPromptTemplate = ChatPromptTemplate.fromMessages([
    ["system", "You are a travel agent specialized in planning activities for tourists."],
    ["user", "Plan a three day itenerary for a trip to {city}."],
]);

var template = await chatPromptTemplate.invoke({ city: "Paris" });
console.log(template);

response = await chat.invoke(template);
console.log(response.content);
