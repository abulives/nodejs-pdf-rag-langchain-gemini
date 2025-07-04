const fs = require('fs');
require('dotenv').config();
const pdfParse = require('pdf-parse');
const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
const { RecursiveCharacterTextSplitter } = require('@langchain/textsplitters');
const { ChatPromptTemplate } = require("@langchain/core/prompts");
const { createRetrievalChain } = require("langchain/chains/retrieval");
const { createStuffDocumentsChain } = require("langchain/chains/combine_documents");
const { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } = require('@langchain/google-genai');

const apiKey = process.env.GOOGLE_API_KEY;
const embeddings = new GoogleGenerativeAIEmbeddings({ model: "models/embedding-001",apiKey });

const llm = new ChatGoogleGenerativeAI({
    model: 'gemini-2.0-flash',
    apiKey,
    temperature: 0.3,
});

const getPdfText = async (files) => {
    try {
        let text = '';
        for (const file of files) {
            const dataBuffer = fs.readFileSync(file.path);
            const pdfData = await pdfParse(dataBuffer);
            text += pdfData.text;
        }
        return text;
    } catch (error) {
        throw error;
    }
};

const getTextChunks = async (text) => {
    try {
        const splitter = new RecursiveCharacterTextSplitter({
            chunk_size: 10000,
            chunk_overlap: 1000,
        });
        const chunks = await splitter.splitText(text);
        return chunks; // list of strings
    } catch (error) {
        throw error;
    }
};

const getVectorStore = async (text_chunks) => {
    try {
        const texts = text_chunks;
        const metadata = texts.map((_, index) => ({ id: index + 1 }));

        const vectorStore = await HNSWLib.fromTexts(text_chunks, metadata, embeddings);
        const vectorname = 'uploaded_vectors';
        await vectorStore.save(`vectors/${vectorname}`);
        return;
    } catch (error) {
        console.error("Error during fromTexts:", error.message);
        throw error;
    }
};

const getConversationalChain = async () => {

    return async ({ input_documents = [], question, vectorStore }) => {
        const context = input_documents.map(doc => doc.pageContent).join('\n');

        const messages = [
            ['system', `Answer the user's questions based on the below context:\n\n{context}, don't provide the wrong answer`],
            ["human", "{input}"]
        ];

        const prompt = ChatPromptTemplate.fromMessages(messages);
        const questionAnswerChain = await createStuffDocumentsChain({ llm, prompt });
        const chain = await createRetrievalChain({
            retriever: vectorStore.asRetriever(),
            combineDocsChain: questionAnswerChain,
        });

        try {
            const response = await chain.invoke({
                input: question
            });
            return { output_text: response.answer };
        } catch (error) {
            console.error("Error calling the model:", error.message);
            throw error;
        }
    };
};

const userInput = async (userQuestion) => {
    try {
        const vectorname = 'uploaded_vectors';
        const vectorStore = await HNSWLib.load(`vectors/${vectorname}`, embeddings);
        const docs = await vectorStore.similaritySearch(userQuestion);
        const chain = await getConversationalChain();
        const response = await chain({
            input_documents: docs,
            question: userQuestion,
            vectorStore
        });
        return response;        
    }catch(err){
        throw new Error(err);
    }
};

module.exports = {
    uploadPDFs: async (files) => {
        try {
            const rawText = await getPdfText(files);
            const textChunks = await getTextChunks(rawText);
            await getVectorStore(textChunks);
            return true;
        }catch(err){
            throw new Error(err);
        }
    },
    askQuestions: async (question) => {
        try {
            const response = await userInput(question);
            return { success: true, response: response.output_text };
        }catch(err){
            console.error('Error in getQuestions:', err);
            return { success: false, error: err.message || 'An error occurred' };
        }
    }
};
