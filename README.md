Question Answering (QA) is a field within natural language processing focused on designing systems that can answer questions. 

In this question answering system will perform two tasks: document retrieval and passage retrieval. Our system will have access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.

algoritum used:

 tf-idf to rank documents by multiply the term frequency of each word by the inverse document frequency 
 : inverse document frequency i.e. how common or rare a word is across documents in a corpus [log(total no of docs/no of times all docs containing the word)]. 
