import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
)
model = ChatHuggingFace(llm = llm)

result = model.invoke("""
    Q: There were 9 computer in server. 4 more server are installed each day from sunday to tuesday. How many computers are now in server room?
    A: There are 3 days from sunday to tuesday. It means 3*4=12 computers were added. there were 9 computer in beginning. so now 
        there is 9+12 = 21 computers. So, final answer is 21 computers.
    Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
""")

print(result.content)