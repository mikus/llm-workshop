#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
    AIMessage,
    HumanMessage,
    SystemMessage,
} from "@langchain/core/messages";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { ChatMessageHistory } from "langchain/stores/message/in_memory";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
  
const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
});


var loader = new CheerioWebBaseLoader(
    "https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/",
    { selector: "article" }
);
var webDocs = await loader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(webDocs);
const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: "text-embedding-3-small",
  });
const vectorStore = await FaissStore.fromDocuments(allSplits, embeddings);
var retriever = vectorStore.asRetriever({
    k: 6,
    searchType: "similarity",
});

/*
### Conversational Q&A

Now that we have a functioning retrieval-based Q&A chain, let's extend it for conversational applications. The main addition is the chat history to persist past interactions.

We'll need to update two things about our existing app:

1. **Prompt:** Update our prompt to support historical messages as an input.
2. **Contextualizing questions:** Add a sub-chain that takes the latest user question and reformulates it in the context of the chat history. This can be thought of simply as building a new "history aware" retriever.
  * Whereas before we had: `query -> retriever`
  * Now we will have: `(query, conversation history) -> LLM -> rephrased query -> retriever`

#### Contextualizing the question
First we'll need to define a sub-chain that takes historical messages and the latest user question, and reformulates the question if it makes reference to any information in the historical information.

We'll use a prompt that includes a `MessagesPlaceholder` variable under the name `chat_history`. This allows us to pass in a list of Messages to the prompt using the `chat_history` input key, and these messages will be inserted after the system message and before the human message containing the latest question.

Note that we leverage a helper function [createHistoryAwareRetriever](https://v03.api.js.langchain.com/functions/langchain.chains_history_aware_retriever.createHistoryAwareRetriever.html) for this step, which manages the case where `chat_history` is empty, and otherwise applies `prompt.pipe(llm).pipe(new StringOutputParser()).pipe(retriever)` in sequence.

`createHistoryAwareRetriever` constructs a chain that accepts keys `input` and `chat_history` as input, and has the same output schema as a retriever.
*/



var contextualizeQSystemPrompt = `Given a chat history and the latest user question
which might reference context in the chat history,
formulate a standalone question which can be understood
without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.
`;

var contextualizeQPrompt = ChatPromptTemplate.fromMessages([
    ["system", contextualizeQSystemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
]);

var historyAwareRetriever = await createHistoryAwareRetriever({
    llm: chat,
    retriever: retriever,
    rephrasePrompt: contextualizeQPrompt
});

/*
This chain prepends a rephrasing of the input query to our retriever, so that the retrieval incorporates the context of the conversation.

Now we can build our full QA chain. This is as simple as updating the retriever to be our new `historyAwareRetriever`.

Again, we will use `createStuffDocumentsChain` to generate a `qaChain`, with input keys `context`, `chat_history`, and `input` -- it accepts the retrieved context alongside the conversation history and query to generate an answer.

We build our final `ragChain` with `createRetrievalChain`. This chain applies the `historyAwareRetriever` and `qaChain` in sequence, retaining intermediate outputs such as the retrieved context for convenience. It has input keys `input` and `chat_history`, and includes `input`, `chat_history`, `context`, and `answer` in its output.
*/
var systemPrompt = `You are an assistant for question-answering tasks.
Use the given context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

{context}
`;

var qaPrompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
]);


var qaChain = await createStuffDocumentsChain({
    llm: chat,
    prompt: qaPrompt,
    outputParser: new StringOutputParser()
});

var ragChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: qaChain,
});

/*
Let's try this. Below we ask a question and a follow-up question that requires contextualization to return a sensible response. Because our chain includes a `chat_history` input, the caller needs to manage the chat history. We can achieve this by appending input and output messages to a list:
*/

var chatHistory = [];

var question = "How can Amazon Kendra be used for developing RAG workflows?";
var aiMsg1 = await ragChain.invoke({input: question, chat_history: chatHistory});

console.log(aiMsg1["answer"]);



chatHistory.push(new HumanMessage(question));
chatHistory.push(new AIMessage(aiMsg1["answer"]));

var secondQuestion = "How does it differ from Amazon Bedrock?";
var aiMsg2 = await ragChain.invoke({input: secondQuestion, chat_history: chatHistory});

console.log(aiMsg2["answer"]);


/*

#### Stateful management of chat history
Here we have gone over how to add application logic for incorporating historical outputs, but we're still manually updating the chat history and inserting it into each input. In a real Q&A application we'll want some way of persisting chat history and some way of automatically inserting and updating it.

For this we can use:

* [BaseChatMessageHistory](https://v03.api.js.langchain.com/classes/_langchain_core.chat_history.BaseChatMessageHistory.html): Store chat history.
* [RunnableWithMessageHistory](https://v03.api.js.langchain.com/classes/_langchain_core.runnables.RunnableWithMessageHistory.html): Wrapper for an chain and a `BaseChatMessageHistory` that handles injecting chat history into inputs and updating it after each invocation.


Below, we implement a simple example of the second option, in which chat histories are stored in a simple dict. LangChain manages memory integrations with [Redis](https://js.langchain.com/docs/integrations/memory/redis/) and other technologies to provide for more robust persistence.

Instances of `RunnableWithMessageHistory` manage the chat history for you. They accept a config with a key (`sessionId` by default) that specifies what conversation history to fetch and prepend to the input, and append the output to the same conversation history. Below is an example:
*/

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

var conversationalRagChain = new RunnableWithMessageHistory({
    runnable: ragChain,
    getMessageHistory,
    inputMessagesKey: "input",
    historyMessagesKey: "chat_history",
    outputMessagesKey: "answer",
});

var response = await conversationalRagChain.invoke(
    { input: "How can Amazon Kendra be used for developing RAG workflows?" },
    { configurable: { sessionId: "abc123" } }
);

console.log(response["answer"]);

var response = await conversationalRagChain.invoke(
    { input: "How does it differ from Amazon Bedrock?" },
    { configurable: { sessionId: "abc123" } }
);

console.log(response["answer"]);

/*
The conversation history can be inspected in the store map:
*/

for (let message of store.get("abc123").messages) {
    let prefix = "Unknown";
    if (message instanceof HumanMessage) {
        prefix = "User";
    } else if (message instanceof AIMessage) {
        prefix = "AI";
    } else if (message instanceof SystemMessage) {
        prefix = "System";
    }
    console.log(`${prefix}: ${message.content}\n`);
}