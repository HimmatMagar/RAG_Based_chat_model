from langchain_community.retrievers import ArxivRetriever

retriver = ArxivRetriever(
    load_max_docs=1,
    load_all_available_meta=True
)

docs = retriver.invoke("Attention is all you need")
print(docs)