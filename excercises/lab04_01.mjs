#!/usr/bin/env node

import "dotenv/config";

import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate, ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { 
    RunnableSequence,
    RunnablePassthrough
} from "@langchain/core/runnables";
import { formatDocumentsAsString } from "langchain/util/document";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
  
const chat = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
});


/*
## Document Loaders

Document loaders are used to load data from external sources as `Documents`. A `Document` is a piece of text (`pageContent`) and associated metadata (`metadata`).
Langchain offers a number of other document loaders and [integrations](https://js.langchain.com/docs/integrations/document_loaders/).

**Try it Yourself!**

Try different document loaders and different prompts for the retrieval chains in the notebook.

Note: Results may not be factually accurate and may be based on false assumptions.
*/

/*
### WebBaseLoader

For this notebook we will load an [AWS blogpost](https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/) on Retrieval Augmented Generation as the external source.

In this case we will use the [CheerioWebBaseLoader](https://js.langchain.com/docs/integrations/document_loaders/web_loaders/web_cheerio/). We can customize which part of the website will be loaded using apropriate selector. In this case only HTML tag `<article>` is relevant, so we will remove all others.
*/


var loader = new CheerioWebBaseLoader(
    "https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/",
    { selector: "article" }
);
var webDocs = await loader.load();

console.log(webDocs[0].metadata);
console.log("------");
console.log(webDocs[0].pageContent.substring(0, 1000));


/*
### PDF Loader
*/
const d2lPath = "data/d2l-textbook.pdf";

var loader = new PDFLoader(d2lPath);
var pdfDocs = await loader.load();

// take 90th page
const page90 = pdfDocs[89];

console.log(page90.metadata);
console.log("------");
console.log(page90.pageContent);


/*
## Document Splitters

Large documents may pose a challenge for RAG as they might not fit into the context window. Document splitting is often performed to split large documents into smaller chunks. This also allows the retriever to select the more relevant chunks from the document instead of feeding the entire data to an LLM. LangChain offers several modules for effectively splitting documents. In this section, we will use the [RecursiveCharacterTextSplitter](https://js.langchain.com/docs/how_to/recursive_text_splitter), which will recursively split the document using common separators like new lines until each chunk is the appropriate size. This is the recommended text splitter for generic text use cases.

A practical idea is to chunk small but fetch large (or an entire doc). Langchain's [ParentDocumentRetriever](https://js.langchain.com/docs/how_to/parent_document_retriever/) allows you to do that. This depends also on how big the context window of your LLM is.
*/

// Use the recursive character splitter
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  
  const allSplits = await textSplitter.splitDocuments(webDocs);
  
  console.log("number of chunks: " + allSplits.length);
  
  // Print a random chunk
  var randomIndex = Math.floor(Math.random() * allSplits.length);
  var randomChunk = allSplits[randomIndex];
  
  console.log(randomChunk.metadata);
  console.log("------");
  console.log(randomChunk.pageContent);

/*
## Vector Store

Since the split chunks need to be retrieved based on semantic relevance, using embeddings serves better than storing the chunks as text. At query time, the query is tranformed into an embeddings and used to find other similar chunk embeddings to retrieve similar chunks.
To store these embeddings for search and retrieval, we use vector stores. In this notebook, we will use [Faiss](https://js.langchain.com/docs/integrations/vectorstores/faiss/) vector database, which is a lightweight vector db than can be run locally. Along with the embeddings, vector databases also store the corresponding text of each chunk.
*/

/*
#### Embedding Model

An embedding model is required to transform the text into vectors represented using embeddings.
We will be using one of the text embeddings models available in Open AI to vectorize the chunks.
*/

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: "text-embedding-3-small",
});

/*
#### Define the vector store
We will now use the embedding model to generate the embeddings and store them in the vector database.
*/
const vectorStore = await FaissStore.fromDocuments(allSplits, embeddings);

/*
## Retrieve

Now let’s write the actual application logic. We want to create a simple application that takes a user question, searches for documents relevant to that question, passes the retrieved documents and initial question to a model, and returns an answer.

First we need to define our logic for searching over documents. LangChain defines a [Retriever](https://js.langchain.com/docs/concepts/retrievers) interface which wraps an index that can return relevant Documents given a string query.

The most common type of Retriever is the [VectorStoreRetriever](https://js.langchain.com/docs/concepts/retrievers/#vector-store), which uses the similarity search capabilities of a vector store to facilitate retrieval. Any `VectorStore` can easily be turned into a `Retriever` with `VectorStore.asRetriever()`:
*/
// Query to retrieve similar chunks
var query = "What options do I have for using LLMs in AWS?";

// Retrieve similar chunks based on relevance. We only retrieve 'k' most similar chunks.
var retriever = vectorStore.asRetriever({
    k: 6,
    searchType: "similarity",
});
var retrievedDocs = await retriever.invoke(query);

console.log("number of retrieved documents: " + retrievedDocs.length);
console.log(retrievedDocs[0].pageContent);

/*
To check how "good" the retrieved documents are in the terms of the used metric ("similarity" in our case) we can query an instance of `VectorStore` directly.
*/
// Retrieve similar chunks based on relevance. We only retrieve 'k' most similar chunks.
var similarChunks = await vectorStore.similaritySearchWithScore(query, 4);

console.log("<table>" +
             "<tr><th>Retrieved Chunks</th><th>Relevance Score</th></tr>" +
             similarChunks.map(([document, score]) => "<tr><td>" + document.pageContent + "</td><td>" + score + "</td></tr>").join("") +
             "</table>"
);

/*
## Generate

Let’s put it all together into a chain that takes a question, retrieves relevant documents, constructs a prompt, passes that to a model, and parses the output.
*/


/*
### Retrival for Q&A</a>

Let's build a Q&A application with a retriever. The retriever returns the chunks from a document based on the relevance with the query. We will examine how using a retriever improves the quality of response by comparing the RAG solution with the vanilla LLM responses.
*/

var qaTemplate = `Use the given context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

Context: {context}

Question: {question}

Answer:
`;

// Define the prompt template for Q&A
var qaPromptTemplate = PromptTemplate.fromTemplate(qaTemplate);

// Define the Q&A chain
var qaChain = RunnableSequence.from([
    {
        context: retriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough()
    },
    qaPromptTemplate,
    chat,
    new StringOutputParser()
]);

// Perform retrieval Q&A
var qaResponse = await qaChain.invoke("List some options for using LLMs in AWS?");

// Format and print in Mardown
console.log(qaResponse);

/*
---

Let's compare the retrieval Q&A response against a vanilla LLM response.
*/
var llmTemplate = `Answer the question below.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

Question: {question}

Answer:
`;

// Prompt template without context added
var llmPromptTemplate = PromptTemplate.fromTemplate(llmTemplate);

// Use the LLM without retriever for response
var llmResponse = await llmPromptTemplate.pipe(chat).invoke({question: "List some options for using LLMs in AWS?"});

console.log(llmResponse.content);

/*
#### Built-in chains
If preferred, LangChain includes convenience functions that implement the above functionality. We compose two functions:

* [createStuffDocumentsChain](https://v03.api.js.langchain.com/functions/langchain.chains_combine_documents.createStuffDocumentsChain.html) specifies how retrieved context is fed into a prompt and LLM. In this case, we will "stuff" the contents into the prompt -- i.e., we will include all retrieved context without any summarization or other processing. It largely implements our above `qaChain`, with input keys `context` and `question` -- it generates an answer using retrieved context and query.
* [createRetrievalChain](https://v03.api.js.langchain.com/functions/langchain.chains_retrieval.createRetrievalChain.html) adds the retrieval step and propagates the retrieved context through the chain, providing it alongside the final answer. It has input key `input`, and includes `input`, `context`, and `answer` in its output.
*/

var systemPrompt = `You are an assistant for question-answering tasks.
Use the given context to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.

{context}
`;

var qaPromptTemplate = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    ["human", "{input}"],
]);

var qaChain = await createStuffDocumentsChain({
    llm: chat,
    prompt: qaPromptTemplate,
    outputParser: new StringOutputParser()
});

var ragChain = await createRetrievalChain({
    retriever: retriever,
    combineDocsChain: qaChain,
});

var response = await ragChain.invoke({input: "List some options for using LLMs in AWS?"})

console.log(response["answer"]);

/*
#### Returning resources

Often in Q&A applications it's important to show users the sources that were used to generate the answer. LangChain's built-in `createRetrievalChain` will propagate retrieved source documents through to the output in the `context` key:
*/
for (let document of response["context"]) {
    console.log(document);
    console.log();
}
