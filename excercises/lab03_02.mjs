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


// The first prompt for the sequential chain
var screenwriterPrompt = PromptTemplate.fromTemplate(`
    You are a screenwriter. Given the title of the movie, it is your job to write a synopsis for that movie.
    Title: {title}
    Screenwriter: This is the synopsis of the above movie:
`);

// The second prompt for the sequential chain
var movieCriticPrompt = PromptTemplate.fromTemplate(`
    You are a movie critic from IMDB. Given the synopsis of the movie, it is your job to write a review of that movie. Be precise.
    Synopsis: {synopsis}
    Review from a IMDB movie critic:
`);

// The third prompt for the sequential chain
var targetDemographicPrompt = PromptTemplate.fromTemplate(`
    Based on the critic review, suggest the target demographic for the movie. Be precise.
    Critic Review: {review}
    Target demographics:
`);

// The fourth prompt for the sequential chain
var socialMediaManagerPrompt = PromptTemplate.fromTemplate(`
    You are a social media manager for a production company. You need to write a short social media post that appeals to the given target demographic given the movie critic review.
    The social media post should mention the rating if it is more than three.
    Target demographic: {targetDemographic}
    Critic review: {review}
    Social media manager: This is the social media post:
`);

var chatParser = chat.pipe(parser);

// global storage for intermediate responses
var responses = {};

// helper function to store intermediate responses
const inspector = (response) => {
  Object.assign(responses, response);
  return response
};

// Define chains for each prompt template
var screenwriterChain = screenwriterPrompt.pipe(chatParser).pipe({synopsis: new RunnablePassthrough()}).pipe(inspector);
var movieCriticChain = movieCriticPrompt.pipe(chatParser).pipe({review: new RunnablePassthrough()}).pipe(inspector);
var targetDemographicChain = targetDemographicPrompt.pipe(chatParser).pipe({targetDemographic: new RunnablePassthrough()}).pipe(inspector);
var socialMediaManagerChain = socialMediaManagerPrompt.pipe(chatParser);

// Combine the four chains together
var combinedChain = screenwriterChain
                        .pipe(movieCriticChain)
                        .pipe(RunnablePassthrough.assign({targetDemographic: targetDemographicChain}))
                        .pipe(socialMediaManagerChain);

// Run the combined chain
var response = await combinedChain.invoke({"title": "Godfather"});

console.log(response);

// Format the responses and print
for (let [key, value] of Object.entries(responses)) {
    console.log("------");
    console.log(`${key}: ${value}\n`);
}
